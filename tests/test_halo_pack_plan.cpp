// Host verification of packed halo gather/scatter index math (single-step and fused PackBatch).
// The CUDA/HIP kernels in src/halo_pack.cu forward to the same host+device functions tested here.
// No GPU, no MPI. Broader plan/geometry coverage lives in test_dense_distributed_host.cpp.

#include <carbon/detail/halo_pack_plan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace hp = carbon::detail::halo_pack;
using carbon::Axis;
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
  const hp::PackBatch batch = hp::make_pack_batch(steps);
  CHECK(batch.num_steps == 2U);
  CHECK(batch.total_z == cl_lo.size() + cl_hi.size());
  CHECK(batch.z_prefix[0] == 0U);
  CHECK(batch.z_prefix[1] == static_cast<uint32_t>(cl_lo.size()));
  CHECK(batch.z_prefix[2] == batch.total_z);
  const hp::PackLaunchShape ls = hp::pack_batch_launch_shape(batch);
  CHECK(ls.grid_z == batch.total_z);
  CHECK(ls.grid_x == hp::pack_launch_shape(steps[0].args).grid_x);
  CHECK(ls.grid_y == hp::pack_launch_shape(steps[0].args).grid_y);

  hp::host_gather_batch<uint32_t>(field.data(), stage_batch.data(), batch);
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
  const hp::PackBatch sbatch = hp::make_pack_batch(scatters);
  hp::host_scatter_batch<uint32_t>(stage_batch.data(), out_batch.data(), sbatch);
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
  CHECK(steps.size() <= hp::kMaxPackBatchSteps);

  std::vector<uint32_t> stage_seq(cl.size() * box.volume(), 0U);
  std::vector<uint32_t> stage_batch(stage_seq.size(), 0U);
  for (const auto& s : steps) {
    hp::host_gather<uint32_t>(field.data(), stage_seq.data() + s.stage_offset, s.args);
  }
  hp::host_gather_batch<uint32_t>(field.data(), stage_batch.data(), hp::make_pack_batch(steps));
  CHECK(stage_batch == stage_seq);
}

auto
decode_prefix() -> void {
  g_ctx = "decode_prefix";
  hp::PackStep steps[3]{};
  steps[0].args.num_comps = 3;
  steps[1].args.num_comps = 1;
  steps[2].args.num_comps = 2;
  steps[0].args.row = 1;
  steps[0].args.ext_x = 1;
  steps[0].args.ext_y = 1;
  steps[0].args.ext_z = 1;
  steps[1].args = steps[0].args;
  steps[1].args.num_comps = 1;
  steps[2].args = steps[0].args;
  steps[2].args.num_comps = 2;
  const hp::PackBatch b = hp::make_pack_batch(steps, 3);
  CHECK(b.total_z == 6U);
  for (uint32_t bz = 0; bz < b.total_z; ++bz) {
    uint32_t step = 99;
    uint32_t local = 99;
    CHECK(hp::pack_batch_decode(b, bz, step, local));
    if (bz < 3) {
      CHECK(step == 0U);
      CHECK(local == bz);
    } else if (bz < 4) {
      CHECK(step == 1U);
      CHECK(local == 0U);
    } else {
      CHECK(step == 2U);
      CHECK(local == bz - 4U);
    }
  }
  uint32_t step = 0;
  uint32_t local = 0;
  CHECK(!hp::pack_batch_decode(b, 6, step, local));
}

}  // namespace

auto
main() -> int {
  single_step_roundtrip();
  fused_batch_matches_sequential();
  batch_chunking_over_max_comps();
  decode_prefix();
  std::printf("OK\n");
  return 0;
}
