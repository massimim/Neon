#pragma once

/**
 * Single-launch Boundary mapping for Pencil spans (@c halo.x == 0, @c halo.y > 0).
 *
 * The Boundary shell of a pencil is four faces: Z-low / Z-high over the full XY extent, and
 * Y-low / Y-high over the full X extent and the middle Z range (same partition as
 * @ref Span::boundary_windows; the X faces are empty because @c halo.x == 0). Instead of one
 * launch per face, one launch covers all of them: every @c blockIdx.z slice is one layer of one
 * face, and x / y are plain @c blockIdx * blockDim + threadIdx. On the Z faces the second
 * thread axis walks y; on the Y faces it walks z. Face selection is uniform per block and
 * needs no per-thread division.
 *
 * @code
 * grid = (ceil(n.x / bx), ceil(max(Z-face rows, Y-face rows) / by), lo.z + hi.z + lo.y + hi.y)
 *
 * s = blockIdx.z
 * s <  lo.z           -> Z-low : (cx, cy, s)                       rows = n.y
 * (s -= lo.z) < hi.z  -> Z-high: (cx, cy, n.z - hi.z + s)          rows = n.y
 * (s -= hi.z) < lo.y  -> Y-low : (cx, s, lo.z + cy)                rows = mz
 * else s -= lo.y      -> Y-high: (cx, n.y - hi.y + s, lo.z + cy)   rows = mz
 * @endcode
 *
 * Cells are owned-local (add @c Span::halo() for the buffer @ref Idx). Band thicknesses come
 * from @ref Span::shell_bands, so chunks thinner than @c 2 * halo are clamped exactly as in
 * @ref Span::boundary_windows. When @c mz == 0 the Y faces are empty and get no slices.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>

#include <cstdint>

namespace carbon::detail::dense_distributed {

/** Owned extent and Boundary band thicknesses of a Pencil span, in 32-bit cells. */
struct PencilBoundaryLayout {
  int32_t nx{0};
  int32_t ny{0};
  int32_t nz{0};
  int32_t lo_z{0};
  int32_t hi_z{0};
  int32_t lo_y{0};
  int32_t hi_y{0};
  /** Z range covered by the Y faces: @c nz - lo_z - hi_z. */
  int32_t mz{0};
};

[[nodiscard]] CARBON_HD constexpr auto
pencil_boundary_layout(const Span& span) noexcept -> PencilBoundaryLayout {
  const int64_3d n = span.logic_extent();
  const ShellBands b = span.shell_bands();
  PencilBoundaryLayout l;
  l.nx = static_cast<int32_t>(n.x);
  l.ny = static_cast<int32_t>(n.y);
  l.nz = static_cast<int32_t>(n.z);
  l.lo_z = static_cast<int32_t>(b.lo.z);
  l.hi_z = static_cast<int32_t>(b.hi.z);
  l.lo_y = static_cast<int32_t>(b.lo.y);
  l.hi_y = static_cast<int32_t>(b.hi.y);
  l.mz = static_cast<int32_t>(n.z - b.lo.z - b.hi.z);
  return l;
}

/** Number of @c blockIdx.z slices: one per layer of each non-empty face. */
[[nodiscard]] CARBON_HD constexpr auto
pencil_boundary_slices(const PencilBoundaryLayout& l) noexcept -> int32_t {
  return l.lo_z + l.hi_z + (l.mz > 0 ? l.lo_y + l.hi_y : 0);
}

/** True when the Boundary launch of @p span on @p block uses the single pencil launch. */
[[nodiscard]] constexpr auto
uses_pencil_boundary_launch(const Span& span,
                            ::carbon::LaunchShape block) noexcept -> bool {
  return span.storage_kind() == StorageKind::Pencil && block.z == 1U;
}

/** Launch grid (in blocks) of the single pencil Boundary launch; all-zero when empty. */
[[nodiscard]] constexpr auto
pencil_boundary_grid(const PencilBoundaryLayout& l,
                     ::carbon::LaunchShape block) noexcept -> ::carbon::LaunchShape {
  const int32_t slices = pencil_boundary_slices(l);
  const int32_t z_rows = (l.lo_z + l.hi_z) > 0 ? l.ny : 0;
  const int32_t y_rows = (l.mz > 0 && (l.lo_y + l.hi_y) > 0) ? l.mz : 0;
  const int32_t rows = z_rows > y_rows ? z_rows : y_rows;
  if (slices <= 0 || rows <= 0 || l.nx <= 0 || block.x == 0U || block.y == 0U) {
    return ::carbon::LaunchShape{0U, 0U, 0U};
  }
  const auto ceil_div = [](int32_t n, std::uint32_t d) -> std::uint32_t {
    return (static_cast<std::uint32_t>(n) + d - 1U) / d;
  };
  return ::carbon::LaunchShape{ceil_div(l.nx, block.x), ceil_div(rows, block.y),
                               static_cast<std::uint32_t>(slices)};
}

/**
 * Owned-local cell of thread (@p thread_x, @p thread_y) in block (@p block_x, @p block_y,
 * @p block_z) of the pencil Boundary launch with block width @p dim_x / height @p dim_y.
 * Returns false (and leaves @p cell unspecified) for threads past the end of their face.
 */
[[nodiscard]] CARBON_HD constexpr auto
pencil_boundary_cell(const PencilBoundaryLayout& l,
                     std::uint32_t block_x,
                     std::uint32_t block_y,
                     std::uint32_t block_z,
                     std::uint32_t thread_x,
                     std::uint32_t thread_y,
                     std::uint32_t dim_x,
                     std::uint32_t dim_y,
                     int64_3d& cell) noexcept -> bool {
  const auto cx = static_cast<int32_t>(block_x * dim_x + thread_x);
  const auto cy = static_cast<int32_t>(block_y * dim_y + thread_y);
  auto s = static_cast<int32_t>(block_z);
  int32_t rows = l.ny;
  if (s < l.lo_z) {
    cell = int64_3d{cx, cy, s};
  } else if ((s -= l.lo_z) < l.hi_z) {
    cell = int64_3d{cx, cy, l.nz - l.hi_z + s};
  } else if ((s -= l.hi_z) < l.lo_y) {
    rows = l.mz;
    cell = int64_3d{cx, s, l.lo_z + cy};
  } else {
    s -= l.lo_y;
    rows = l.mz;
    cell = int64_3d{cx, l.ny - l.hi_y + s, l.lo_z + cy};
  }
  return cx < l.nx && cy < rows;
}

}  // namespace carbon::detail::dense_distributed
