#pragma once

/**
 * Single-launch Boundary mapping for Brick spans (@c halo.x > 0).
 *
 * The Boundary shell of a brick is six faces: Z-low / Z-high over the full XY extent, Y-low /
 * Y-high over the full X extent and the middle Z range, X-low / X-high over the middle Y and
 * middle Z ranges (same partition as @ref Span::boundary_windows). Instead of one launch per
 * face, one launch covers all of them: every @c blockIdx.z slice is one layer of one face, and
 * x / y are plain @c blockIdx * blockDim + threadIdx. On the Z faces the thread axes walk (x, y),
 * on the Y faces (x, z) and on the X faces (y, z), so no face is launched one cell wide. Face
 * selection is uniform per block and needs no per-thread division.
 *
 * @code
 * grid = (ceil(max cols / bx), ceil(max rows / by), slices)   over the faces that get slices
 *
 * s = blockIdx.z
 * s <  lo.z           -> Z-low : (cx, cy, s)                          cols n.x, rows n.y
 * (s -= lo.z) < hi.z  -> Z-high: (cx, cy, n.z - hi.z + s)             cols n.x, rows n.y
 * (s -= hi.z) < lo.y  -> Y-low : (cx, s, lo.z + cy)                   cols n.x, rows mz
 * (s -= lo.y) < hi.y  -> Y-high: (cx, n.y - hi.y + s, lo.z + cy)      cols n.x, rows mz
 * (s -= hi.y) < lo.x  -> X-low : (s, lo.y + cx, lo.z + cy)            cols my,  rows mz
 * else s -= lo.x      -> X-high: (n.x - hi.x + s, lo.y + cx, lo.z + cy)
 * @endcode
 *
 * Cells are owned-local (add @c Span::halo() for the buffer @ref Idx). Band thicknesses come
 * from @ref Span::shell_bands, so chunks thinner than @c 2 * halo are clamped exactly as in
 * @ref Span::boundary_windows. The Y faces get slices only when @c mz > 0 and the X faces only
 * when @c mz > 0 and @c my > 0.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>

#include <cstdint>

namespace carbon::detail::dense_distributed {

/** Owned extent and Boundary band thicknesses of a Brick span, in 32-bit cells. */
struct BrickBoundaryLayout {
  int32_t nx{0};
  int32_t ny{0};
  int32_t nz{0};
  int32_t lo_z{0};
  int32_t hi_z{0};
  int32_t lo_y{0};
  int32_t hi_y{0};
  int32_t lo_x{0};
  int32_t hi_x{0};
  /** Y range covered by the X faces: @c ny - lo_y - hi_y. */
  int32_t my{0};
  /** Z range covered by the Y and X faces: @c nz - lo_z - hi_z. */
  int32_t mz{0};
};

[[nodiscard]] CARBON_HD constexpr auto
brick_boundary_layout(const Span& span) noexcept -> BrickBoundaryLayout {
  const int64_3d n = span.logic_extent();
  const ShellBands b = span.shell_bands();
  BrickBoundaryLayout l;
  l.nx = static_cast<int32_t>(n.x);
  l.ny = static_cast<int32_t>(n.y);
  l.nz = static_cast<int32_t>(n.z);
  l.lo_z = static_cast<int32_t>(b.lo.z);
  l.hi_z = static_cast<int32_t>(b.hi.z);
  l.lo_y = static_cast<int32_t>(b.lo.y);
  l.hi_y = static_cast<int32_t>(b.hi.y);
  l.lo_x = static_cast<int32_t>(b.lo.x);
  l.hi_x = static_cast<int32_t>(b.hi.x);
  l.my = static_cast<int32_t>(n.y - b.lo.y - b.hi.y);
  l.mz = static_cast<int32_t>(n.z - b.lo.z - b.hi.z);
  return l;
}

/** Slices of the Y faces: none when the middle Z range is empty. */
[[nodiscard]] CARBON_HD constexpr auto
brick_boundary_y_slices(const BrickBoundaryLayout& l) noexcept -> int32_t {
  return l.mz > 0 ? l.lo_y + l.hi_y : 0;
}

/** Slices of the X faces: none when the middle Y or Z range is empty. */
[[nodiscard]] CARBON_HD constexpr auto
brick_boundary_x_slices(const BrickBoundaryLayout& l) noexcept -> int32_t {
  return (l.mz > 0 && l.my > 0) ? l.lo_x + l.hi_x : 0;
}

/** Number of @c blockIdx.z slices: one per layer of each non-empty face. */
[[nodiscard]] CARBON_HD constexpr auto
brick_boundary_slices(const BrickBoundaryLayout& l) noexcept -> int32_t {
  return l.lo_z + l.hi_z + brick_boundary_y_slices(l) + brick_boundary_x_slices(l);
}

/** True when the Boundary launch of @p span on @p block uses the single brick launch. */
[[nodiscard]] constexpr auto
uses_brick_boundary_launch(const Span& span,
                           ::carbon::LaunchShape block) noexcept -> bool {
  return span.storage_kind() == StorageKind::Brick && block.z == 1U;
}

/** Launch grid (in blocks) of the single brick Boundary launch; all-zero when empty. */
[[nodiscard]] constexpr auto
brick_boundary_grid(const BrickBoundaryLayout& l,
                    ::carbon::LaunchShape block) noexcept -> ::carbon::LaunchShape {
  int32_t cols = 0;
  int32_t rows = 0;
  const auto cover = [&cols, &rows](int32_t slices, int32_t c, int32_t r) {
    if (slices > 0) {
      cols = c > cols ? c : cols;
      rows = r > rows ? r : rows;
    }
  };
  cover(l.lo_z + l.hi_z, l.nx, l.ny);
  cover(brick_boundary_y_slices(l), l.nx, l.mz);
  cover(brick_boundary_x_slices(l), l.my, l.mz);
  const int32_t slices = brick_boundary_slices(l);
  if (slices <= 0 || cols <= 0 || rows <= 0 || block.x == 0U || block.y == 0U) {
    return ::carbon::LaunchShape{0U, 0U, 0U};
  }
  const auto ceil_div = [](int32_t n, std::uint32_t d) -> std::uint32_t {
    return (static_cast<std::uint32_t>(n) + d - 1U) / d;
  };
  return ::carbon::LaunchShape{ceil_div(cols, block.x), ceil_div(rows, block.y),
                               static_cast<std::uint32_t>(slices)};
}

/**
 * Owned-local cell of thread (@p thread_x, @p thread_y) in block (@p block_x, @p block_y,
 * @p block_z) of the brick Boundary launch with block width @p dim_x / height @p dim_y.
 * Returns false (and leaves @p cell unspecified) for threads past the end of their face.
 */
[[nodiscard]] CARBON_HD constexpr auto
brick_boundary_cell(const BrickBoundaryLayout& l,
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
  int32_t cols = l.nx;
  int32_t rows = l.ny;
  if (s < l.lo_z) {
    cell = int64_3d{cx, cy, s};
  } else if ((s -= l.lo_z) < l.hi_z) {
    cell = int64_3d{cx, cy, l.nz - l.hi_z + s};
  } else if ((s -= l.hi_z) < l.lo_y) {
    rows = l.mz;
    cell = int64_3d{cx, s, l.lo_z + cy};
  } else if ((s -= l.lo_y) < l.hi_y) {
    rows = l.mz;
    cell = int64_3d{cx, l.ny - l.hi_y + s, l.lo_z + cy};
  } else if ((s -= l.hi_y) < l.lo_x) {
    cols = l.my;
    rows = l.mz;
    cell = int64_3d{s, l.lo_y + cx, l.lo_z + cy};
  } else {
    s -= l.lo_x;
    cols = l.my;
    rows = l.mz;
    cell = int64_3d{l.nx - l.hi_x + s, l.lo_y + cx, l.lo_z + cy};
  }
  return cx < cols && cy < rows;
}

}  // namespace carbon::detail::dense_distributed
