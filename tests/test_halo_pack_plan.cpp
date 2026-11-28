// Host verification of packed halo gather/scatter index math (single-step and fused PackTable),
// and of the Pencil and Brick Direct plans against the staged plan and the global truth on an
// emulated multi-rank world.
// The CUDA/HIP kernels in src/halo_pack.cu forward to the same host+device functions tested here.
// No GPU, no MPI. Broader plan/geometry coverage lives in test_dense_distributed_host.cpp.

#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/halo_pack_plan.h>
#include <carbon/process_grid.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

namespace hp = carbon::detail::halo_pack;
using carbon::Axis;
using carbon::BoundaryModeDistributed;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::detail::halo_distributed_build::AxisSide;
using hp::AxisGeometry;

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n  context: %s\n", #cond, __FILE__, \
                   __LINE__, g_ctx.c_str());                                              \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

std::string g_ctx;

auto
fill_field(std::vector<uint32_t>& field) -> void {
  for (size_t i = 0; i < field.size(); ++i) {
    field[i] = static_cast<uint32_t>(i * 2654435761ULL + 17U);
  }
}

auto
check_gather_box(const std::vector<uint32_t>& field,
                 const std::vector<uint32_t>& stage,
                 const hp::PackBox& box,
                 const std::vector<int>& comps,
                 const AxisGeometry& g,
                 int64_t cs,
                 size_t stage_base) -> void {
  size_t n = stage_base;
  for (const int c : comps) {
    for (int64_t z = 0; z < box.ext_z; ++z) {
      for (int64_t y = 0; y < box.ext_y; ++y) {
        for (int64_t x = 0; x < box.ext_x; ++x) {
          const uint32_t expect = field[static_cast<size_t>(
              c * cs + (box.lo_z + z) * g.pitch + (box.lo_y + y) * g.haloed.x + box.lo_x + x)];
          CHECK(stage[n] == expect);
          ++n;
        }
      }
    }
  }
}

auto
single_step_roundtrip() -> void {
  g_ctx = "single_step_roundtrip";
  AxisGeometry g;
  g.owned = {5, 6, 7};
  g.halo = {2, 1, 2};
  g.haloed = {9, 8, 11};
  g.pitch = g.haloed.x * g.haloed.y + 3;
  const int comps = 4;
  const int64_t cs = g.pitch * g.haloed.z;
  std::vector<uint32_t> field(static_cast<size_t>(cs) * comps);
  fill_field(field);
  std::vector<int> cl = {0, 2, 3};
  const hp::PackBox box = hp::pack_face_box(g, Axis::Y, AxisSide::Lower, 1, g.haloed.x - 1);
  std::vector<hp::PackStep> steps;
  hp::detail_plan::append_steps(steps, box, cl, g, cs, /*region_offset=*/0);
  CHECK(steps.size() == 1U);

  std::vector<uint32_t> stage(cl.size() * box.volume(), 0U);
  hp::host_gather<uint32_t>(field.data(), stage.data(), steps[0].args);
  check_gather_box(field, stage, box, cl, g, cs, 0);

  std::vector<uint32_t> out(field.size(), 0U);
  hp::host_scatter<uint32_t>(stage.data(), out.data(), steps[0].args);
  size_t written = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != 0U) {
      CHECK(out[i] == field[i]);
      ++written;
    }
  }
  CHECK(written <= stage.size());
}

auto
fused_batch_matches_sequential() -> void {
  g_ctx = "fused_batch_matches_sequential";
  AxisGeometry g;
  g.owned = {8, 9, 10};
  g.halo = {1, 2, 1};
  g.haloed = {10, 13, 12};
  g.pitch = g.haloed.x * g.haloed.y;
  const int comps = 5;
  const int64_t cs = g.pitch * g.haloed.z;
  std::vector<uint32_t> field(static_cast<size_t>(cs) * comps);
  fill_field(field);

  // Two Y faces (Lower + Upper owned faces) with different component lists, staged back to back.
  std::vector<hp::PackStep> steps;
  const hp::PackBox lo = hp::pack_face_box(g, Axis::Y, AxisSide::Lower, 0, g.haloed.x);
  const hp::PackBox hi = hp::pack_face_box(g, Axis::Y, AxisSide::Upper, 0, g.haloed.x);
  CHECK(lo.volume() == hi.volume());
  const std::vector<int> cl_lo = {0, 1, 4};
  const std::vector<int> cl_hi = {1, 2};
  hp::detail_plan::append_steps(steps, lo, cl_lo, g, cs, /*region_offset=*/0);
  const size_t mid = cl_lo.size() * lo.volume();
  hp::detail_plan::append_steps(steps, hi, cl_hi, g, cs, /*region_offset=*/mid);
  CHECK(steps.size() == 2U);

  const size_t stage_n = mid + cl_hi.size() * hi.volume();
  std::vector<uint32_t> stage_seq(stage_n, 0U);
  std::vector<uint32_t> stage_batch(stage_n, 0U);
  for (const auto& s : steps) {
    hp::host_gather<uint32_t>(field.data(), stage_seq.data() + s.stage_offset, s.args);
  }
  const hp::PackTableImage image = hp::make_pack_table_image(steps);
  const hp::PackTable batch = image.view(image.bytes.data());
  CHECK(batch.num_steps == 2U);
  const hp::PackLaunchShape s0 = hp::pack_launch_shape(steps[0].args);
  const hp::PackLaunchShape s1 = hp::pack_launch_shape(steps[1].args);
  CHECK(batch.block_prefix[0] == 0U);
  CHECK(batch.block_prefix[1] == s0.grid_x * s0.grid_y * s0.grid_z);
  CHECK(batch.total_blocks == batch.block_prefix[1] + s1.grid_x * s1.grid_y * s1.grid_z);
  CHECK(batch.block_prefix[2] == batch.total_blocks);
  CHECK(batch.stage_offsets[1] == mid);
  const hp::PackLaunchShape ls = hp::pack_table_launch_shape(batch);
  CHECK(ls.grid_x == batch.total_blocks);
  CHECK(ls.grid_y == 1U);
  CHECK(ls.grid_z == 1U);

  hp::host_gather_table<uint32_t>(field.data(), stage_batch.data(), batch);
  CHECK(stage_batch == stage_seq);

  std::vector<uint32_t> out_seq(field.size(), 0U);
  std::vector<uint32_t> out_batch(field.size(), 0U);
  // Scatter into the halo boxes (not the owned faces).
  std::vector<hp::PackStep> scatters;
  const hp::PackBox halo_lo = hp::pack_halo_box(g, Axis::Y, AxisSide::Lower, 0, g.haloed.x);
  const hp::PackBox halo_hi = hp::pack_halo_box(g, Axis::Y, AxisSide::Upper, 0, g.haloed.x);
  hp::detail_plan::append_steps(scatters, halo_lo, cl_lo, g, cs, /*region_offset=*/0);
  hp::detail_plan::append_steps(scatters, halo_hi, cl_hi, g, cs, /*region_offset=*/mid);
  for (const auto& s : scatters) {
    hp::host_scatter<uint32_t>(stage_seq.data() + s.stage_offset, out_seq.data(), s.args);
  }
  const hp::PackTableImage simage = hp::make_pack_table_image(scatters);
  hp::host_scatter_table<uint32_t>(stage_batch.data(), out_batch.data(),
                                   simage.view(simage.bytes.data()));
  CHECK(out_batch == out_seq);
}

auto
batch_chunking_over_max_comps() -> void {
  g_ctx = "batch_chunking_over_max_comps";
  AxisGeometry g;
  g.owned = {4, 5, 6};
  g.halo = {1, 1, 1};
  g.haloed = {6, 7, 8};
  g.pitch = g.haloed.x * g.haloed.y;
  const int comps = 40;  // > kMaxPackComps => two PackSteps for one face
  const int64_t cs = g.pitch * g.haloed.z;
  std::vector<uint32_t> field(static_cast<size_t>(cs) * comps);
  fill_field(field);
  std::vector<int> cl(comps);
  for (int c = 0; c < comps; ++c) {
    cl[static_cast<size_t>(c)] = c;
  }
  const hp::PackBox box = hp::pack_face_box(g, Axis::Z, AxisSide::Upper, 0, g.haloed.x);
  std::vector<hp::PackStep> steps;
  hp::detail_plan::append_steps(steps, box, cl, g, cs, 0);
  CHECK(steps.size() == 2U);

  std::vector<uint32_t> stage_seq(cl.size() * box.volume(), 0U);
  std::vector<uint32_t> stage_batch(stage_seq.size(), 0U);
  for (const auto& s : steps) {
    hp::host_gather<uint32_t>(field.data(), stage_seq.data() + s.stage_offset, s.args);
  }
  const hp::PackTableImage image = hp::make_pack_table_image(steps);
  hp::host_gather_table<uint32_t>(field.data(), stage_batch.data(), image.view(image.bytes.data()));
  CHECK(stage_batch == stage_seq);
}

// Every block of the 1D table grid decodes to one (step, bx, by, bz) of that step's own grid, every
// block of every step's grid is reached exactly once, and the chunked prefix count picks the same
// step as a linear scan of block_prefix.
auto
check_decode(const std::vector<hp::PackStep>& steps) -> void {
  const hp::PackTableImage image = hp::make_pack_table_image(steps);
  CHECK(image.bytes.size() % hp::kPackTableAlign == 0U);
  CHECK(image.stage_offsets_at % hp::kPackTableAlign == 0U);
  CHECK(image.grid_x_at % hp::kPackTableAlign == 0U);
  CHECK(image.grid_y_at % hp::kPackTableAlign == 0U);
  CHECK(image.block_prefix_at % hp::kPackTableAlign == 0U);
  const hp::PackTable t = image.view(image.bytes.data());
  CHECK(t.num_steps == steps.size());
  std::vector<std::vector<int>> seen(steps.size());
  for (size_t i = 0; i < steps.size(); ++i) {
    const hp::PackLaunchShape s = hp::pack_launch_shape(steps[i].args);
    seen[i].assign(static_cast<size_t>(s.grid_x) * s.grid_y * s.grid_z, 0);
    CHECK(t.stage_offsets[i] == steps[i].stage_offset);
  }
  for (uint32_t block = 0; block < t.total_blocks; ++block) {
    uint32_t step = 99;
    uint32_t bx = 0;
    uint32_t by = 0;
    uint32_t bz = 0;
    CHECK(hp::pack_table_decode(t, block, /*lane=*/0U, step, bx, by, bz));
    uint32_t linear = 0U;
    while (linear + 1U < t.num_steps && block >= t.block_prefix[linear + 1U]) {
      ++linear;
    }
    CHECK(step == linear);
    const hp::PackLaunchShape s = hp::pack_launch_shape(steps[step].args);
    CHECK(bx < s.grid_x);
    CHECK(by < s.grid_y);
    CHECK(bz < s.grid_z);
    ++seen[step][(static_cast<size_t>(bz) * s.grid_y + by) * s.grid_x + bx];
  }
  for (const auto& v : seen) {
    for (const int n : v) {
      CHECK(n == 1);
    }
  }
  uint32_t step = 0;
  uint32_t bx = 0;
  uint32_t by = 0;
  uint32_t bz = 0;
  CHECK(!hp::pack_table_decode(t, t.total_blocks, /*lane=*/0U, step, bx, by, bz));
}

auto
make_shape_step(uint32_t row,
                uint32_t z,
                uint32_t comps,
                size_t stage_offset) -> hp::PackStep {
  hp::PackStep step{};
  step.args.ext_x = row;
  step.args.ext_y = 1U;
  step.args.ext_z = z;
  step.args.row = row;
  step.args.num_comps = comps;
  step.stage_offset = stage_offset;
  return step;
}

auto
decode_blocks() -> void {
  g_ctx = "decode_blocks";
  // Wide row / short z, narrow row / long z, single block.
  check_decode({make_shape_step(600U, 1U, 3U, 0U), make_shape_step(3U, 70U, 2U, 1800U),
                make_shape_step(1U, 1U, 1U, 2220U)});
  check_decode({make_shape_step(1U, 1U, 1U, 0U)});
}

// More steps than the old by-value batch held (16), and more than one kPackPrefixChunk: the table
// is one launch whatever its length.
auto
decode_blocks_many_steps() -> void {
  g_ctx = "decode_blocks_many_steps";
  for (const uint32_t n : {31U, 32U, 33U, 40U, 64U, 70U}) {
    std::vector<hp::PackStep> steps;
    size_t offset = 0U;
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t row = 1U + (i * 97U) % 700U;
      const uint32_t z = 1U + (i * 13U) % 9U;
      const uint32_t comps = 1U + i % 4U;
      steps.push_back(make_shape_step(row, z, comps, offset));
      offset += static_cast<size_t>(row) * z * comps;
    }
    check_decode(steps);
  }
}

// ---- Pencil Direct on an emulated multi-rank world -----------------------------------------

constexpr uint32_t kHaloSentinel = 0xDEADBEEFU;

struct Rank {
  int32_3d coords{};
  AxisGeometry g{};
  int64_3d offset{};  ///< global coordinate of owned cell (0, 0, 0)
  std::vector<uint32_t> field{};
  std::vector<uint32_t> stage{};
};

using DirActive = std::function<bool(int, int32_t, int32_t)>;

auto
split_extent(int64_t n,
             int32_t parts,
             int32_t i,
             int64_t& lo,
             int64_t& len) -> void {
  const int64_t base = n / parts;
  const int64_t extra = n % parts;
  len = base + (i < extra ? 1 : 0);
  lo = i * base + std::min<int64_t>(i, extra);
}

auto
truth(int c,
      int64_t x,
      int64_t y,
      int64_t z) -> uint32_t {
  const auto h = static_cast<uint64_t>(((c * 131 + x) * 1009 + y) * 4001 + z);
  return static_cast<uint32_t>(h * 2654435761ULL);
}

// Owned cells hold the global truth, every halo cell the sentinel.
auto
make_world(const carbon::ProcessGrid& pg,
           BoundaryModeDistributed bc,
           int64_3d global,
           int32_t r,
           int comps) -> std::vector<Rank> {
  const int32_3d dims = pg.dims();
  const int64_t hx = dims.x > 1 || carbon::is_periodic(bc, Axis::X) ? r : 0;
  const int64_t hy = dims.y > 1 || carbon::is_periodic(bc, Axis::Y) ? r : 0;
  const int64_t hz = dims.z > 1 || carbon::is_periodic(bc, Axis::Z) ? r : 0;
  std::vector<Rank> world(static_cast<size_t>(pg.world_size()));
  for (int32_t rank = 0; rank < pg.world_size(); ++rank) {
    Rank& w = world[static_cast<size_t>(rank)];
    w.coords = pg.coords_of(rank);
    int64_t lo_x = 0;
    int64_t nx = 0;
    int64_t lo_y = 0;
    int64_t ny = 0;
    int64_t lo_z = 0;
    int64_t nz = 0;
    split_extent(global.x, dims.x, w.coords.x, lo_x, nx);
    split_extent(global.y, dims.y, w.coords.y, lo_y, ny);
    split_extent(global.z, dims.z, w.coords.z, lo_z, nz);
    w.g.owned = {nx, ny, nz};
    w.g.halo = {hx, hy, hz};
    w.g.haloed = {nx + 2 * hx, ny + 2 * hy, nz + 2 * hz};
    w.g.pitch = w.g.haloed.x * w.g.haloed.y;
    w.offset = {lo_x, lo_y, lo_z};
    const int64_t cs = w.g.pitch * w.g.haloed.z;
    w.field.assign(static_cast<size_t>(cs * comps), kHaloSentinel);
    for (int c = 0; c < comps; ++c) {
      for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
          for (int64_t x = 0; x < nx; ++x) {
            w.field[static_cast<size_t>(c * cs + (z + hz) * w.g.pitch + (y + hy) * w.g.haloed.x +
                                        x + hx)] = truth(c, lo_x + x, lo_y + y, lo_z + z);
          }
        }
      }
    }
  }
  return world;
}

// One fused launch over the pass's table, as HaloDistributed::run_pack_table_.
auto
host_run_steps(const std::vector<hp::PackStep>& steps,
               std::vector<uint32_t>& field,
               std::vector<uint32_t>& stage,
               bool gather) -> void {
  if (steps.empty()) {
    return;
  }
  const hp::PackTableImage image = hp::make_pack_table_image(steps);
  const hp::PackTable t = image.view(image.bytes.data());
  if (gather) {
    hp::host_gather_table<uint32_t>(field.data(), stage.data(), t);
  } else {
    hp::host_scatter_table<uint32_t>(stage.data(), field.data(), t);
  }
}

// One packed plan per rank: fused gathers, message copies matched by (peer, key), fused scatters.
auto
run_plans(std::vector<Rank>& world,
          const std::vector<hp::AxisPackPlan>& plans) -> void {
  for (size_t i = 0; i < world.size(); ++i) {
    world[i].stage.assign(plans[i].stage_elems, 0U);
    host_run_steps(plans[i].gathers, world[i].field, world[i].stage, /*gather=*/true);
  }
  for (size_t i = 0; i < world.size(); ++i) {
    for (const hp::PackMsg& recv : plans[i].msgs) {
      if (recv.op != hp::MsgOp::Recv) {
        continue;
      }
      const auto src = static_cast<size_t>(recv.peer);
      CHECK(src != i);
      int matches = 0;
      for (const hp::PackMsg& send : plans[src].msgs) {
        if (send.op == hp::MsgOp::Send && send.peer == static_cast<int>(i) &&
            send.key == recv.key) {
          CHECK(send.count == recv.count);
          std::copy_n(world[src].stage.begin() + static_cast<std::ptrdiff_t>(send.first),
                      send.count, world[i].stage.begin() + static_cast<std::ptrdiff_t>(recv.first));
          ++matches;
        }
      }
      CHECK(matches == 1);
    }
    // Every send has exactly one matching recv on its peer.
    for (const hp::PackMsg& send : plans[i].msgs) {
      if (send.op != hp::MsgOp::Send) {
        continue;
      }
      int matches = 0;
      for (const hp::PackMsg& recv : plans[static_cast<size_t>(send.peer)].msgs) {
        matches +=
            recv.op == hp::MsgOp::Recv && recv.peer == static_cast<int>(i) && recv.key == send.key
                ? 1
                : 0;
      }
      CHECK(matches == 1);
    }
  }
  for (size_t i = 0; i < world.size(); ++i) {
    host_run_steps(plans[i].scatters, world[i].field, world[i].stage, /*gather=*/false);
  }
}

auto
wrap(int64_t v,
     int64_t n,
     bool periodic,
     bool& ok) -> int64_t {
  if (v >= 0 && v < n) {
    return v;
  }
  if (!periodic) {
    ok = false;
    return v;
  }
  return ((v % n) + n) % n;
}

using DirActive3 = std::function<bool(int, int32_t, int32_t, int32_t)>;

auto
region(int64_t v,
       int64_t halo,
       int64_t owned) -> int32_t {
  return v < halo ? -1 : (v >= halo + owned ? 1 : 0);
}

// Owned cells unchanged; a halo cell in region (kx, ky, kz) holds the global value at its wrapped
// coordinate when that coordinate exists and @p active(c, kx, ky, kz), else the sentinel.
auto
check_truth3(const std::vector<Rank>& world,
             BoundaryModeDistributed bc,
             int64_3d global,
             int comps,
             const DirActive3& active) -> void {
  for (const Rank& w : world) {
    const int64_t cs = w.g.pitch * w.g.haloed.z;
    for (int c = 0; c < comps; ++c) {
      for (int64_t z = 0; z < w.g.haloed.z; ++z) {
        for (int64_t y = 0; y < w.g.haloed.y; ++y) {
          for (int64_t x = 0; x < w.g.haloed.x; ++x) {
            const int32_t kx = region(x, w.g.halo.x, w.g.owned.x);
            const int32_t ky = region(y, w.g.halo.y, w.g.owned.y);
            const int32_t kz = region(z, w.g.halo.z, w.g.owned.z);
            bool ok = true;
            const int64_t gx =
                wrap(w.offset.x + x - w.g.halo.x, global.x, carbon::is_periodic(bc, Axis::X), ok);
            const int64_t gy =
                wrap(w.offset.y + y - w.g.halo.y, global.y, carbon::is_periodic(bc, Axis::Y), ok);
            const int64_t gz =
                wrap(w.offset.z + z - w.g.halo.z, global.z, carbon::is_periodic(bc, Axis::Z), ok);
            const bool filled = ok && ((kx == 0 && ky == 0 && kz == 0) || active(c, kx, ky, kz));
            const uint32_t got =
                w.field[static_cast<size_t>(c * cs + z * w.g.pitch + y * w.g.haloed.x + x)];
            CHECK(got == (filled ? truth(c, gx, gy, gz) : kHaloSentinel));
          }
        }
      }
    }
  }
}

auto
check_truth(const std::vector<Rank>& world,
            BoundaryModeDistributed bc,
            int64_3d global,
            int comps,
            const DirActive& active) -> void {
  check_truth3(world, bc, global, comps, [&](int c, int32_t, int32_t ky, int32_t kz) {
    return active(c, ky, kz);
  });
}

auto
direct_plans(const carbon::ProcessGrid& pg,
             BoundaryModeDistributed bc,
             const std::vector<Rank>& world,
             int comps,
             const DirActive& send,
             const DirActive& recv) -> std::vector<hp::AxisPackPlan> {
  std::vector<hp::AxisPackPlan> plans;
  for (size_t i = 0; i < world.size(); ++i) {
    const int64_t cs = world[i].g.pitch * world[i].g.haloed.z;
    plans.push_back(hp::build_pencil_direct_pack_plan(pg, bc, world[i].coords,
                                                      static_cast<int32_t>(i), world[i].g, comps,
                                                      cs, send, recv, /*stage_base=*/0));
  }
  return plans;
}

auto
pencil_direct_vs_staged() -> void {
  struct Case {
    int32_t py;
    int32_t pz;
    BoundaryModeDistributed bc;
    int64_3d global;
    int32_t r;
  };
  const BoundaryModeDistributed open = BoundaryModeDistributed::Open;
  const BoundaryModeDistributed py = BoundaryModeDistributed::PeriodicY;
  const BoundaryModeDistributed pz = BoundaryModeDistributed::PeriodicZ;
  const Case cases[] = {
      {2, 2, open, {5, 7, 9}, 1},    {2, 4, open, {4, 9, 13}, 1},   {2, 4, py | pz, {3, 8, 12}, 1},
      {3, 2, py, {4, 10, 7}, 2},     {1, 3, py, {5, 6, 11}, 1},     {2, 1, pz, {4, 9, 5}, 1},
      {1, 1, py | pz, {3, 5, 6}, 1}, {1, 2, py | pz, {3, 5, 8}, 2}, {2, 2, py | pz, {3, 6, 6}, 1},
      {4, 2, pz, {2, 17, 9}, 2},
  };
  const int comps = 3;
  const DirActive all = [](int, int32_t, int32_t) {
    return true;
  };
  for (const Case& k : cases) {
    g_ctx = "pencil direct " + std::to_string(k.py) + "x" + std::to_string(k.pz) +
            " bc=" + std::to_string(static_cast<int>(k.bc)) + " r=" + std::to_string(k.r);
    const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(k.py * k.pz, 1, k.py, k.pz);
    std::vector<Rank> direct = make_world(pg, k.bc, k.global, k.r, comps);
    std::vector<Rank> staged = direct;

    run_plans(direct, direct_plans(pg, k.bc, direct, comps, all, all));

    for (const Axis axis : {Axis::Y, Axis::Z}) {
      std::vector<hp::AxisPackPlan> plans;
      for (size_t i = 0; i < staged.size(); ++i) {
        const int64_t cs = staged[i].g.pitch * staged[i].g.haloed.z;
        plans.push_back(hp::build_axis_pack_plan(
            pg, k.bc, staged[i].coords, static_cast<int32_t>(i), staged[i].g, axis, comps, cs,
            [](int, Axis, AxisSide) {
              return true;
            },
            [](int, Axis, AxisSide) {
              return true;
            },
            /*stage_base=*/0));
      }
      run_plans(staged, plans);
    }

    for (size_t i = 0; i < direct.size(); ++i) {
      CHECK(direct[i].field == staged[i].field);
    }
    check_truth(direct, k.bc, k.global, comps, all);
  }
}

// Lattice filters: a halo cell in region (ky, kz) only carries the populations streaming into it
// from the neighbour there; every other population keeps the sentinel.
auto
pencil_direct_lattice() -> void {
  // D3Q19 (vy, vz) velocity components; x does not matter on a pencil.
  const int vel[19][2] = {{0, 0},  {0, 0},  {0, 0},   {1, 0},  {-1, 0}, {0, 1},  {0, -1},
                          {1, 0},  {-1, 0}, {1, 0},   {-1, 0}, {0, 1},  {0, -1}, {0, 1},
                          {0, -1}, {1, 1},  {-1, -1}, {1, -1}, {-1, 1}};
  const int comps = 19;
  const int64_3d global{3, 6, 12};
  for (const bool bidirectional : {false, true}) {
    const auto crosses = [bidirectional](int32_t sign, int v) {
      if (sign == 0) {
        return true;
      }
      if (bidirectional) {
        return v != 0;
      }
      return sign > 0 ? v > 0 : v < 0;
    };
    const DirActive send = [&](int c, int32_t dy, int32_t dz) {
      return crosses(dy, vel[c][0]) && crosses(dz, vel[c][1]);
    };
    const DirActive recv = [&](int c, int32_t dy, int32_t dz) {
      return crosses(-dy, vel[c][0]) && crosses(-dz, vel[c][1]);
    };
    for (const BoundaryModeDistributed bc :
         {BoundaryModeDistributed::Open,
          BoundaryModeDistributed::PeriodicY | BoundaryModeDistributed::PeriodicZ}) {
      g_ctx = std::string("pencil direct lattice bidirectional=") + (bidirectional ? "1" : "0") +
              " bc=" + std::to_string(static_cast<int>(bc));
      const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(8, 1, 2, 4);
      std::vector<Rank> world = make_world(pg, bc, global, 1, comps);
      run_plans(world, direct_plans(pg, bc, world, comps, send, recv));
      check_truth(world, bc, global, comps, recv);
    }
  }
}

// One rank, Y and Z periodic: all eight directions are self wraps, so no messages.
auto
pencil_direct_self_only() -> void {
  g_ctx = "pencil direct self only";
  const BoundaryModeDistributed bc =
      BoundaryModeDistributed::PeriodicY | BoundaryModeDistributed::PeriodicZ;
  const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(1, 1, 1, 1);
  const std::vector<Rank> world = make_world(pg, bc, {4, 5, 6}, 1, 2);
  const DirActive all = [](int, int32_t, int32_t) {
    return true;
  };
  const auto plans = direct_plans(pg, bc, world, 2, all, all);
  CHECK(plans[0].msgs.empty());
  CHECK(plans[0].gathers.size() == 8U);
  CHECK(plans[0].scatters.size() == 8U);
}

// Pencil Direct refuses spans that are not pencils.
auto
pencil_direct_rejects_non_pencil() -> void {
  g_ctx = "pencil direct rejects non pencil";
  const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(2, 1, 1, 2);
  AxisGeometry g;
  g.owned = {4, 4, 4};
  g.halo = {0, 0, 1};
  g.haloed = {4, 4, 6};
  g.pitch = 16;
  const DirActive all = [](int, int32_t, int32_t) {
    return true;
  };
  bool threw = false;
  try {
    (void)hp::build_pencil_direct_pack_plan(pg, BoundaryModeDistributed::Open, {0, 0, 0}, 0, g, 1,
                                            g.pitch * g.haloed.z, all, all, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

// ---- Brick Direct ----------------------------------------------------------------------------

auto
brick_direct_plans(const carbon::ProcessGrid& pg,
                   BoundaryModeDistributed bc,
                   const std::vector<Rank>& world,
                   int comps,
                   const DirActive3& send,
                   const DirActive3& recv) -> std::vector<hp::AxisPackPlan> {
  std::vector<hp::AxisPackPlan> plans;
  for (size_t i = 0; i < world.size(); ++i) {
    const int64_t cs = world[i].g.pitch * world[i].g.haloed.z;
    plans.push_back(hp::build_brick_direct_pack_plan(pg, bc, world[i].coords,
                                                     static_cast<int32_t>(i), world[i].g, comps, cs,
                                                     send, recv, /*stage_base=*/0));
  }
  return plans;
}

auto
run_staged_xyz(const carbon::ProcessGrid& pg,
               BoundaryModeDistributed bc,
               std::vector<Rank>& world,
               int comps) -> void {
  for (const Axis axis : {Axis::X, Axis::Y, Axis::Z}) {
    std::vector<hp::AxisPackPlan> plans;
    for (size_t i = 0; i < world.size(); ++i) {
      const int64_t cs = world[i].g.pitch * world[i].g.haloed.z;
      plans.push_back(hp::build_axis_pack_plan(
          pg, bc, world[i].coords, static_cast<int32_t>(i), world[i].g, axis, comps, cs,
          [](int, Axis, AxisSide) {
            return true;
          },
          [](int, Axis, AxisSide) {
            return true;
          },
          /*stage_base=*/0));
    }
    run_plans(world, plans);
  }
}

auto
brick_direct_vs_staged() -> void {
  struct Case {
    int32_t px;
    int32_t py;
    int32_t pz;
    BoundaryModeDistributed bc;
    int64_3d global;
    int32_t r;
  };
  const BoundaryModeDistributed open = BoundaryModeDistributed::Open;
  const BoundaryModeDistributed px = BoundaryModeDistributed::PeriodicX;
  const BoundaryModeDistributed py = BoundaryModeDistributed::PeriodicY;
  const BoundaryModeDistributed pz = BoundaryModeDistributed::PeriodicZ;
  const Case cases[] = {
      {2, 2, 2, open, {7, 9, 11}, 1},        {2, 2, 1, open, {8, 9, 5}, 1},
      {2, 1, 2, px | pz, {6, 5, 8}, 2},      {3, 2, 2, px | py | pz, {9, 7, 6}, 1},
      {2, 2, 2, px | py | pz, {5, 6, 7}, 1}, {1, 2, 2, px, {5, 8, 9}, 2},
      {2, 3, 2, py, {9, 10, 7}, 2},          {1, 1, 1, px, {4, 5, 6}, 1},
      {2, 2, 2, px, {5, 7, 9}, 2},           {3, 3, 3, open, {9, 10, 11}, 1},
      {1, 1, 1, px | py | pz, {4, 5, 6}, 2}, {2, 2, 2, open, {6, 6, 40}, 1},
  };
  const int comps = 3;
  const DirActive3 all = [](int, int32_t, int32_t, int32_t) {
    return true;
  };
  for (const Case& k : cases) {
    g_ctx = "brick direct " + std::to_string(k.px) + "x" + std::to_string(k.py) + "x" +
            std::to_string(k.pz) + " bc=" + std::to_string(static_cast<int>(k.bc)) +
            " r=" + std::to_string(k.r);
    const carbon::ProcessGrid pg =
        carbon::ProcessGrid::custom(k.px * k.py * k.pz, k.px, k.py, k.pz);
    std::vector<Rank> direct = make_world(pg, k.bc, k.global, k.r, comps);
    CHECK(direct[0].g.halo.x > 0);
    std::vector<Rank> staged = direct;

    run_plans(direct, brick_direct_plans(pg, k.bc, direct, comps, all, all));
    run_staged_xyz(pg, k.bc, staged, comps);

    for (size_t i = 0; i < direct.size(); ++i) {
      CHECK(direct[i].field == staged[i].field);
    }
    check_truth3(direct, k.bc, k.global, comps, all);
  }
}

// Lattice filters: a halo cell in region (kx, ky, kz) only carries the populations streaming into
// it from the neighbour there; every other population keeps the sentinel.
auto
brick_direct_lattice() -> void {
  // D3Q19 then the eight D3Q27 corner velocities.
  const int vel[27][3] = {
      {0, 0, 0},  {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {0, -1, 0}, {0, 0, 1},   {0, 0, -1},
      {1, 1, 0},  {-1, -1, 0}, {1, -1, 0},  {-1, 1, 0},  {1, 0, 1},  {-1, 0, -1}, {1, 0, -1},
      {-1, 0, 1}, {0, 1, 1},   {0, -1, -1}, {0, 1, -1},  {0, -1, 1}, {1, 1, 1},   {-1, -1, -1},
      {1, 1, -1}, {-1, -1, 1}, {1, -1, 1},  {-1, 1, -1}, {-1, 1, 1}, {1, -1, -1},
  };
  const BoundaryModeDistributed all_periodic = BoundaryModeDistributed::PeriodicX |
                                               BoundaryModeDistributed::PeriodicY |
                                               BoundaryModeDistributed::PeriodicZ;
  for (const int comps : {19, 27}) {
    for (const bool bidirectional : {false, true}) {
      const auto crosses = [bidirectional](int32_t sign, int v) {
        if (sign == 0) {
          return true;
        }
        if (bidirectional) {
          return v != 0;
        }
        return sign > 0 ? v > 0 : v < 0;
      };
      const DirActive3 send = [&](int c, int32_t dx, int32_t dy, int32_t dz) {
        return crosses(dx, vel[c][0]) && crosses(dy, vel[c][1]) && crosses(dz, vel[c][2]);
      };
      const DirActive3 recv = [&](int c, int32_t dx, int32_t dy, int32_t dz) {
        return send(c, -dx, -dy, -dz);
      };
      for (const BoundaryModeDistributed bc : {BoundaryModeDistributed::Open, all_periodic}) {
        for (const int32_t n : {2, 3}) {
          g_ctx = "brick direct lattice Q" + std::to_string(comps) +
                  " bidirectional=" + (bidirectional ? "1" : "0") +
                  " bc=" + std::to_string(static_cast<int>(bc)) + " grid=" + std::to_string(n);
          const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(n * n * n, n, n, n);
          const int64_3d global{7, 8, 9};
          std::vector<Rank> world = make_world(pg, bc, global, 1, comps);
          const auto plans = brick_direct_plans(pg, bc, world, comps, send, recv);
          run_plans(world, plans);
          check_truth3(world, bc, global, comps, recv);

          // The LBM benchmark case: 2x2x2 with walls, every rank a process-grid corner with three
          // face, three edge and one corner neighbour; D3Q19 has no corner population.
          if (n == 2 && bc == BoundaryModeDistributed::Open) {
            const size_t dirs = comps == 19 ? 6U : 7U;
            for (const hp::AxisPackPlan& p : plans) {
              size_t sends = 0;
              for (const hp::PackMsg& m : p.msgs) {
                sends += m.op == hp::MsgOp::Send ? 1U : 0U;
              }
              CHECK(sends == dirs);
              CHECK(p.msgs.size() == 2U * dirs);
              CHECK(p.gathers.size() == dirs);
              CHECK(p.scatters.size() == dirs);
            }
          }
        }
      }
    }
  }
}

// Edges along z (r x r x long) are packed with y and z swapped: the step still covers exactly the
// box cells, and the row now spans the long axis.
auto
brick_direct_thin_edge_swap() -> void {
  g_ctx = "brick direct thin edge swap";
  AxisGeometry g;
  g.owned = {6, 7, 40};
  g.halo = {1, 2, 1};
  g.haloed = {8, 11, 42};
  g.pitch = g.haloed.x * g.haloed.y;
  const int comps = 2;
  const int64_t cs = g.pitch * g.haloed.z;
  std::vector<uint32_t> field(static_cast<size_t>(cs) * comps);
  fill_field(field);
  const std::vector<int> cl = {0, 1};
  for (const bool send : {true, false}) {
    const hp::DirectDir d{1, -1, 0};
    const hp::PackBox box =
        send ? hp::brick_direct_send_box(g, d) : hp::brick_direct_recv_box(g, d);
    CHECK(box.ext_x == 1 && box.ext_y == 2 && box.ext_z == 40);
    CHECK(hp::pack_swap_yz(box));
    std::vector<hp::PackStep> steps;
    hp::detail_plan::append_oriented_steps(steps, box, cl, g, cs, 0);
    CHECK(steps.size() == 1U);
    CHECK(steps[0].args.ext_y == 40U && steps[0].args.ext_z == 2U && steps[0].args.row == 40U);

    std::vector<uint32_t> stage(cl.size() * box.volume(), 0U);
    hp::host_gather<uint32_t>(field.data(), stage.data(), steps[0].args);
    std::vector<uint32_t> out(field.size(), 0U);
    hp::host_scatter<uint32_t>(stage.data(), out.data(), steps[0].args);
    for (int c = 0; c < comps; ++c) {
      for (int64_t z = 0; z < g.haloed.z; ++z) {
        for (int64_t y = 0; y < g.haloed.y; ++y) {
          for (int64_t x = 0; x < g.haloed.x; ++x) {
            const bool inside = x >= box.lo_x && x < box.lo_x + box.ext_x && y >= box.lo_y &&
                                y < box.lo_y + box.ext_y && z >= box.lo_z &&
                                z < box.lo_z + box.ext_z;
            const auto i = static_cast<size_t>(c * cs + z * g.pitch + y * g.haloed.x + x);
            CHECK(out[i] == (inside ? field[i] : 0U));
          }
        }
      }
    }
  }
  // A face with a wide row keeps the default orientation.
  CHECK(!hp::pack_swap_yz(hp::brick_direct_send_box(g, hp::DirectDir{0, 0, 1})));
}

// One rank, X, Y and Z periodic: all 26 directions are self wraps, so no messages.
auto
brick_direct_self_only() -> void {
  g_ctx = "brick direct self only";
  const BoundaryModeDistributed bc = BoundaryModeDistributed::PeriodicX |
                                     BoundaryModeDistributed::PeriodicY |
                                     BoundaryModeDistributed::PeriodicZ;
  const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(1, 1, 1, 1);
  const std::vector<Rank> world = make_world(pg, bc, {4, 5, 6}, 1, 2);
  const DirActive3 all = [](int, int32_t, int32_t, int32_t) {
    return true;
  };
  const auto plans = brick_direct_plans(pg, bc, world, 2, all, all);
  CHECK(plans[0].msgs.empty());
  CHECK(plans[0].gathers.size() == 26U);
  CHECK(plans[0].scatters.size() == 26U);
}

// Brick Direct refuses spans without an x halo.
auto
brick_direct_rejects_no_x_halo() -> void {
  g_ctx = "brick direct rejects no x halo";
  const carbon::ProcessGrid pg = carbon::ProcessGrid::custom(4, 1, 2, 2);
  AxisGeometry g;
  g.owned = {4, 4, 4};
  g.halo = {0, 1, 1};
  g.haloed = {4, 6, 6};
  g.pitch = 24;
  const DirActive3 all = [](int, int32_t, int32_t, int32_t) {
    return true;
  };
  bool threw = false;
  try {
    (void)hp::build_brick_direct_pack_plan(pg, BoundaryModeDistributed::Open, {0, 0, 0}, 0, g, 1,
                                           g.pitch * g.haloed.z, all, all, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

auto
main() -> int {
  single_step_roundtrip();
  fused_batch_matches_sequential();
  batch_chunking_over_max_comps();
  decode_blocks();
  decode_blocks_many_steps();
  pencil_direct_vs_staged();
  pencil_direct_lattice();
  pencil_direct_self_only();
  pencil_direct_rejects_non_pencil();
  brick_direct_vs_staged();
  brick_direct_lattice();
  brick_direct_thin_edge_swap();
  brick_direct_self_only();
  brick_direct_rejects_no_x_halo();
  std::printf("OK\n");
  return 0;
}
