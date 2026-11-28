// DenseDistributed Brick Boundary launch on one GPU: Internal + Boundary == Standard.
//
// A Brick span (halo.x > 0) runs its whole Boundary view in a single launch whose blockIdx.z
// selects the face and halo layer (see detail/dense_distributed/brick_boundary.h). One GPU is
// enough to reach that path: a 1x1x1 process grid with PeriodicX gets an x halo and is a Brick
// span without MPI (halo (r, 0, r); with PeriodicY too, (r, r, r)).
//
// For each case a kernel adds 1 to a per-cell counter. Run once on Standard and once on Internal
// then Boundary; afterwards every owned cell must be exactly 1 and every halo cell still 0 in both
// fields, so each owned cell is visited exactly once and nothing outside the owned region is
// written. Cases cover radius 1 and 2 stencils, no y halo (no Y faces), chunks thinner than
// 2 * halo in x and in z (the Y and X faces are then empty), X faces wider than the x extent, and
// block overrides: a narrower z == 1 block (single launch) and a z == 2 block (falls back to one
// launch per Boundary window). The reported gpu_grid(Boundary) must match the single launch's
// grid.

#include <carbon/boundary_mode_distributed.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_distributed.h>
#include <carbon/detail/dense_distributed/brick_boundary.h>
#include <carbon/detail/gpu.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using carbon::BoundaryModeDistributed;
using carbon::DataView;
using carbon::DenseDistributed;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::LaunchShape;
namespace dd = carbon::detail::dense_distributed;

struct Case {
  const char* label;
  int64_3d global;
  int radius;
  BoundaryModeDistributed mode;
  std::optional<LaunchShape> block;
};

[[nodiscard]] auto
stencil_of_radius(int r) -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<int32_3d>{{r, 0, 0}, {-r, 0, 0}, {0, r, 0}, {0, -r, 0}, {0, 0, r}, {0, 0, -r}},
      /*remove_center=*/true);
}

[[nodiscard]] auto
run_case(carbon::System& sys,
         const Case& c) -> bool {
  const std::string label = c.label;
  try {
    DenseDistributed grid(sys, c.global, carbon::ProcessGrid::custom(1, 1, 1, 1),
                          stencil_of_radius(c.radius), c.mode, carbon::float32_3d{}, 1.0F, c.block);
    const auto& span = grid.spans()[0];
    if (span.storage_kind() != dd::StorageKind::Brick) {
      CARBON_LOG_ERROR("test_dense_distributed_brick_boundary[{}]: span is not Brick", label);
      return false;
    }
    const LaunchShape block = grid.gpu_block(DataView::Boundary)[0];
    if (dd::uses_brick_boundary_launch(span, block)) {
      const LaunchShape want = dd::brick_boundary_grid(dd::brick_boundary_layout(span), block);
      const LaunchShape got = grid.gpu_grid(DataView::Boundary)[0];
      if (got != want) {
        CARBON_LOG_ERROR(
            "test_dense_distributed_brick_boundary[{}]: gpu_grid(Boundary)=({},{},{}) != "
            "brick grid ({},{},{})",
            label, got.x, got.y, got.z, want.x, want.y, want.z);
        return false;
      }
    }

    auto count_std = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "count_std");
    auto count_split = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "count_split");
    for (auto* f : {&count_std, &count_split}) {
      auto& h = f->cpu_buffers()[0];
      for (size_t i = 0; i < h.size(); ++i) {
        h.data()[i] = 0;
      }
    }

    carbon::Stream stream = sys.stream();
    carbon::Transfer<int>(sys, count_std.cpu_buffers(), count_std.gpu_buffers())(stream);
    carbon::Transfer<int>(sys, count_split.cpu_buffers(), count_split.gpu_buffers())(stream);
    stream.wait();

    using Span = DenseDistributed::Span;
    using Idx = DenseDistributed::Idx;
    const auto make_counter = [&grid](DenseDistributed::Field<int, 1>& f) {
      return grid.make_kernel(
          [&f](carbon::KernelLoader& loader) {
            const auto p = loader(f);
            return [p] CARBON_HOST_DEVICE(Span s, Idx idx) {
              p.write(s, idx, 0, p.read(s, idx, 0) + 1);
            };
          },
          /*shared_mem=*/0U);
    };
    auto k_std = make_counter(count_std);
    auto k_split = make_counter(count_split);
    k_std.launch(stream, DataView::Standard);
    k_split.launch(stream, DataView::Internal);
    k_split.launch(stream, DataView::Boundary);
    stream.wait();

    carbon::Transfer<int>(sys, count_std.gpu_buffers(), count_std.cpu_buffers())(stream);
    carbon::Transfer<int>(sys, count_split.gpu_buffers(), count_split.cpu_buffers())(stream);
    stream.wait();

    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_3d haloed = span.haloed_extent();
    const int* hs = count_std.cpu_buffers()[0].data();
    const int* hp = count_split.cpu_buffers()[0].data();
    size_t bad = 0;
    size_t owned_cells = 0;
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const bool is_owned = x >= halo.x && x < halo.x + owned.x && y >= halo.y &&
                                y < halo.y + owned.y && z >= halo.z && z < halo.z + owned.z;
          const int want = is_owned ? 1 : 0;
          const int64_t off = span.linear_offset(span.make_idx(x, y, z), 0);
          if (hs[off] != want || hp[off] != want) {
            if (bad < 4U) {
              CARBON_LOG_ERROR(
                  "test_dense_distributed_brick_boundary[{}]: buffer ({},{},{}) owned={} "
                  "Standard={} Internal+Boundary={}",
                  label, x, y, z, is_owned, hs[off], hp[off]);
            }
            ++bad;
          }
          owned_cells += is_owned ? 1U : 0U;
        }
      }
    }
    if (bad != 0U) {
      CARBON_LOG_ERROR("test_dense_distributed_brick_boundary[{}]: {} wrong cells", label, bad);
      return false;
    }
    CARBON_LOG_INFO(
        "test_dense_distributed_brick_boundary[{}]: OK (owned {}x{}x{}, halo ({},{},{}), "
        "block ({},{},{}), {} owned cells)",
        label, owned.x, owned.y, owned.z, halo.x, halo.y, halo.z, block.x, block.y, block.z,
        owned_cells);
    return true;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_brick_boundary[{}]: {}", label, e.what());
    return false;
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_dense_distributed_brick_boundary: no devices visible");
    return 1;
  }
  std::vector<carbon::detail::Gpu> all = carbon::get_all_devices();
  std::vector<carbon::detail::Gpu> single;
  single.push_back(std::move(all.front()));
  carbon::System sys(std::move(single));

  constexpr auto kX = BoundaryModeDistributed::PeriodicX;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  const Case cases[] = {
      {"r1-periodicX", {70, 37, 21}, 1, kX, std::nullopt},
      {"r1-periodicXY", {70, 37, 21}, 1, kX | kY, std::nullopt},
      {"r1-periodicXYZ", {70, 37, 21}, 1, kX | kY | kZ, std::nullopt},
      {"r2-periodicXY", {70, 37, 21}, 2, kX | kY, std::nullopt},
      {"r2-thin-x", {3, 37, 21}, 2, kX | kY, std::nullopt},
      {"r2-thin-z", {40, 9, 3}, 2, kX | kY, std::nullopt},
      {"r1-wide-x-faces", {20, 70, 40}, 1, kX | kY, std::nullopt},
      {"r1-block-32x4", {70, 37, 21}, 1, kX | kY, LaunchShape{32U, 4U, 1U}},
      {"r1-block-32x4x2-fallback", {70, 37, 21}, 1, kX | kY, LaunchShape{32U, 4U, 2U}},
  };
  int failures = 0;
  for (const Case& c : cases) {
    if (!run_case(sys, c)) {
      ++failures;
    }
  }
  if (failures != 0) {
    CARBON_LOG_ERROR("test_dense_distributed_brick_boundary: {} case(s) failed", failures);
    return 1;
  }
  CARBON_LOG_INFO("test_dense_distributed_brick_boundary: all cases passed");
  return 0;
}
