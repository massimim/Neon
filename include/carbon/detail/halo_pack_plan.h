#pragma once

/**
 * @file halo_pack_plan.h
 * @brief Host-side planning and shared index math for the *packed* multi-axis halo exchange.
 *
 * Why this exists
 * ---------------
 * The run-based multi-axis builder in @c halo_distributed_build.h describes every halo face as a list of
 * contiguous runs. In the x-fastest layout that is one run per (y, z) row for an X face
 * (@c haloed.y * @c haloed.z runs of @c halo.x elements) and one run per z-plane for a Y face.
 * Each run becomes one @c ncclSend / @c ncclRecv (cross-rank) or one async copy (intra). At a
 * 256^3 local block that is ~66k transfers per face per component: the exchange is dominated by
 * per-message overhead, not bandwidth.
 *
 * The packed path instead gathers a face into one contiguous staging region, moves it with ONE
 * message per (axis, side), and scatters it into the halo on the other side:
 *
 *     gather (kernel) -> [intra copies] -> NCCL group (1 Send + 1 Recv per side) -> scatter (kernel)
 *
 * X and Y faces are always packed. A Z face is one contiguous run PER COMPONENT, so it only needs
 * packing when it carries several components (a D3Q19 population field moves 5 populations per
 * face with the lattice filter, 19 without): without packing that is one NCCL message per
 * component per direction, and the messages that cross a node boundary are issued back to back
 * through the network proxy. The caller decides (see HaloDistributed::build_multi_axis_); the plan builder
 * handles Z like any other axis.
 *
 * This header is host-only and has no backend dependency, so the whole plan (extents, component
 * lists, peers, message ordering) and the per-thread index math are unit-testable with a plain
 * C++ compiler (see tests/test_dense_distributed_host.cpp and tests/test_halo_pack_plan.cpp). The
 * CUDA/HIP kernels in src/halo_pack.cu are thin wrappers that forward their launch coordinates to
 * @ref pack_gather_thread / @ref pack_scatter_thread (and the fused @ref pack_gather_table_thread /
 * @ref pack_scatter_table_thread), so the tested code IS the device code.
 *
 * Face extents (design: staged X -> Y -> Z)
 * -----------------------------------------
 *  - X pass : halo.x  x  OWNED y  x  OWNED z
 *  - Y pass : filled-x  x  halo.y  x  OWNED z      ("filled-x" = x range already written by the
 *                                                   X pass; exterior x halos on open axes are
 *                                                   excluded, so they are never overwritten)
 *  - Z pass : filled-x  x  filled-y  x  halo.z      (same rule for y: the X and Y passes have
 *                                                   already filled every halo cell that has a
 *                                                   neighbour, corners included)
 * The run-based path sends the full haloed y/z (resp. z) range instead; those extra rows carry
 * not-yet-filled halo cells (wasted bandwidth) and can overwrite exterior halo cells on end ranks
 * of open axes.
 */

#include <carbon/boundary_mode_distributed.h>
#include <carbon/cuda_annotation.h>
#include <carbon/detail/halo_distributed_build.h>
#include <carbon/process_grid.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__CUDACC__) || defined(__HIPCC__)
#define CARBON_PACK_RESTRICT __restrict__
#else
#define CARBON_PACK_RESTRICT
#endif

namespace carbon::detail::halo_pack {

using halo_distributed_build::AxisGeometry;
using halo_distributed_build::AxisSide;

/** Maximum components packed by one @ref PackStep (its argument struct is fixed-size). */
inline constexpr uint32_t kMaxPackComps = 32U;
/** Threads per block along x for pack/unpack kernels. */
inline constexpr uint32_t kPackBlockX = 256U;

/** Axis-aligned box in *field buffer* coordinates (x fastest, includes halos). */
struct PackBox {
  int64_t lo_x{};
  int64_t lo_y{};
  int64_t lo_z{};
  int64_t ext_x{};
  int64_t ext_y{};
  int64_t ext_z{};

  [[nodiscard]] constexpr auto
  volume() const noexcept -> size_t {
    return static_cast<size_t>(ext_x) * static_cast<size_t>(ext_y) * static_cast<size_t>(ext_z);
  }
};

/**
 * Everything a pack/unpack kernel needs, as a POD passed by value.
 *
 * Field addressing (matches DenseDistributed): element (c, x, y, z) lives at
 *   c * comp_stride + z * pitch + y * haloed_x + x.
 * Staging addressing: plane k (= the k-th packed component) is contiguous and x-fastest:
 *   k * stage_comp_stride + (iz * ext_y + iy) * ext_x + ix,   stage_comp_stride = ext_x*ext_y*ext_z.
 */
struct PackKernelArgs {
  int64_t lo_x{};
  int64_t lo_y{};
  int64_t lo_z{};
  int64_t haloed_x{};
  int64_t pitch{};
  int64_t comp_stride{};
  int64_t stage_comp_stride{};
  uint32_t ext_x{};
  uint32_t ext_y{};
  uint32_t ext_z{};
  uint32_t row{};  ///< ext_x * ext_y (threads along the flattened x/y plane of the box)
  uint32_t num_comps{};
  int32_t comps[kMaxPackComps]{};
};

struct PackLaunchShape {
  uint32_t grid_x{};
  uint32_t grid_y{};
  uint32_t grid_z{};
  uint32_t block_x{};
};

/** Launch geometry: x = flattened (x,y) row, y = grid-stride over z, z = packed component. */
[[nodiscard]] inline auto
pack_launch_shape(const PackKernelArgs& a) noexcept -> PackLaunchShape {
  PackLaunchShape s;
  s.block_x = kPackBlockX;
  s.grid_x = (a.row + kPackBlockX - 1U) / kPackBlockX;
  s.grid_y = std::min<uint32_t>(a.ext_z, 65535U);
  s.grid_z = a.num_comps;
  return s;
}

/**
 * Several @c PackStep launches fused into one kernel. Step @c s owns the contiguous block range
 * @c [block_prefix[s], block_prefix[s+1]) of a 1D grid, laid out as its own single-step grid
 * (@ref pack_launch_shape) flattened component-major: component, then z stride, then row block.
 * Each step keeps its own grid shape, so steps of very different shapes (Direct mixes faces that
 * are wide in the row with faces that are long in z) share a launch without idle blocks.
 *
 * The arrays are not part of the struct: a kernel gets this view (a few pointers and counts) by
 * value and reads the steps from device memory, so one launch covers any number of steps.
 * @ref PackTableImage builds the arrays as one byte image, which HaloDistributed uploads once at
 * construction (host tests view the image in place).
 *
 * @c stage_offsets are element offsets into the caller's staging base pointer (the same values
 * stored on @c PackStep::stage_offset).
 */
struct PackTable {
  const PackKernelArgs* steps{};
  const size_t* stage_offsets{};
  const uint32_t* grid_x{};        ///< row blocks of step s
  const uint32_t* grid_y{};        ///< z grid-stride of step s
  const uint32_t* block_prefix{};  ///< num_steps + 1 entries
  uint32_t num_steps{};
  uint32_t total_blocks{};
};

/** Launch geometry for a fused table: one 1D grid covering every step's blocks. */
[[nodiscard]] inline auto
pack_table_launch_shape(const PackTable& t) noexcept -> PackLaunchShape {
  PackLaunchShape s;
  s.block_x = kPackBlockX;
  s.grid_x = t.total_blocks;
  s.grid_y = 1U;
  s.grid_z = 1U;
  return s;
}

/** Steps of @c block_prefix tested together by @ref pack_prefix_count (one per warp lane). */
inline constexpr uint32_t kPackPrefixChunk = 32U;

/**
 * Number of steps @c s in <tt>[base, base + kPackPrefixChunk)</tt> with
 * <tt>block_prefix[s] <= block</tt>. On CUDA devices each lane of the (full) warp reads one entry
 * and the warp counts with a ballot, so the search is one load instead of a chain of dependent
 * ones; elsewhere one thread reads the chunk (independent loads, same result). Every lane of a warp
 * must call it with the same @p base and @p block.
 */
[[nodiscard]] CARBON_HD inline auto
pack_prefix_count(const PackTable& t,
                  uint32_t base,
                  uint32_t block,
                  uint32_t lane) noexcept -> uint32_t {
#if defined(__CUDA_ARCH__)
  const uint32_t s = base + lane;
  const bool le = s < t.num_steps && t.block_prefix[s] <= block;
  return static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFFU, le)));
#else
  (void)lane;
  const uint32_t end =
      t.num_steps - base < kPackPrefixChunk ? t.num_steps : base + kPackPrefixChunk;
  uint32_t n = 0U;
  for (uint32_t s = base; s < end; ++s) {
    n += t.block_prefix[s] <= block ? 1U : 0U;
  }
  return n;
#endif
}

/**
 * Map the 1D block index @p block to its step and to that step's own (@p bx, @p by, @p bz)
 * block coordinates. Block-uniform: the step is the number of @c block_prefix entries at or below
 * @p block, minus one (@c block_prefix is strictly increasing, since every step has at least one
 * block), counted @ref kPackPrefixChunk at a time (a one-step table skips the search, which would
 * otherwise cost it a dependent load); then two divisions. @p lane is the caller's lane
 * in its warp (see @ref pack_prefix_count); call before any thread of the warp exits.
 */
[[nodiscard]] CARBON_HD inline auto
pack_table_decode(const PackTable& t,
                  uint32_t block,
                  uint32_t lane,
                  uint32_t& step,
                  uint32_t& bx,
                  uint32_t& by,
                  uint32_t& bz) noexcept -> bool {
  if (block >= t.total_blocks || t.num_steps == 0U) {
    return false;
  }
  uint32_t s = 0U;
  if (t.num_steps > 1U) {
    uint32_t count = 0U;
    for (uint32_t base = 0U; base < t.num_steps; base += kPackPrefixChunk) {
      const uint32_t n = pack_prefix_count(t, base, block, lane);
      count += n;
      if (n < kPackPrefixChunk) {
        break;
      }
    }
    s = count - 1U;
  }
  const uint32_t local = block - t.block_prefix[s];
  const uint32_t gx = t.grid_x[s];
  const uint32_t per_comp = gx * t.grid_y[s];
  bz = local / per_comp;
  const uint32_t rem = local - bz * per_comp;
  by = rem / gx;
  bx = rem - by * gx;
  step = s;
  return true;
}

/**
 * One device thread of the gather kernel (field -> staging). Host+device so the same code runs
 * in the CUDA/HIP kernel and in the host unit test.
 */
template <typename W>
CARBON_HD inline void
pack_gather_thread(const W* CARBON_PACK_RESTRICT field,
                   W* CARBON_PACK_RESTRICT stage,
                   const PackKernelArgs& a,
                   uint32_t bx,
                   uint32_t by,
                   uint32_t bz,
                   uint32_t tx,
                   uint32_t bdx,
                   uint32_t gdy) {
  const uint32_t j = bx * bdx + tx;
  if (j >= a.row || bz >= a.num_comps) {
    return;
  }
  const uint32_t iy = j / a.ext_x;
  const uint32_t ix = j - iy * a.ext_x;
  const int64_t fbase = static_cast<int64_t>(a.comps[bz]) * a.comp_stride +
                        (a.lo_y + static_cast<int64_t>(iy)) * a.haloed_x + a.lo_x +
                        static_cast<int64_t>(ix);
  const int64_t sbase = static_cast<int64_t>(bz) * a.stage_comp_stride + static_cast<int64_t>(j);
  for (uint32_t iz = by; iz < a.ext_z; iz += gdy) {
    stage[sbase + static_cast<int64_t>(iz) * static_cast<int64_t>(a.row)] =
        field[fbase + (a.lo_z + static_cast<int64_t>(iz)) * a.pitch];
  }
}

/** One device thread of the scatter kernel (staging -> field). */
template <typename W>
CARBON_HD inline void
pack_scatter_thread(const W* CARBON_PACK_RESTRICT stage,
                    W* CARBON_PACK_RESTRICT field,
                    const PackKernelArgs& a,
                    uint32_t bx,
                    uint32_t by,
                    uint32_t bz,
                    uint32_t tx,
                    uint32_t bdx,
                    uint32_t gdy) {
  const uint32_t j = bx * bdx + tx;
  if (j >= a.row || bz >= a.num_comps) {
    return;
  }
  const uint32_t iy = j / a.ext_x;
  const uint32_t ix = j - iy * a.ext_x;
  const int64_t fbase = static_cast<int64_t>(a.comps[bz]) * a.comp_stride +
                        (a.lo_y + static_cast<int64_t>(iy)) * a.haloed_x + a.lo_x +
                        static_cast<int64_t>(ix);
  const int64_t sbase = static_cast<int64_t>(bz) * a.stage_comp_stride + static_cast<int64_t>(j);
  for (uint32_t iz = by; iz < a.ext_z; iz += gdy) {
    field[fbase + (a.lo_z + static_cast<int64_t>(iz)) * a.pitch] =
        stage[sbase + static_cast<int64_t>(iz) * static_cast<int64_t>(a.row)];
  }
}

/**
 * One device thread of a fused gather: @p block selects the step via @ref pack_table_decode.
 * @p bdx must be a multiple of the warp size (every lane takes part in the step search).
 */
template <typename W>
CARBON_HD inline void
pack_gather_table_thread(const W* CARBON_PACK_RESTRICT field,
                         W* CARBON_PACK_RESTRICT stage_base,
                         const PackTable& t,
                         uint32_t block,
                         uint32_t tx,
                         uint32_t bdx) {
  uint32_t step = 0U;
  uint32_t bx = 0U;
  uint32_t by = 0U;
  uint32_t bz = 0U;
  if (!pack_table_decode(t, block, tx % kPackPrefixChunk, step, bx, by, bz)) {
    return;
  }
  pack_gather_thread<W>(field, stage_base + t.stage_offsets[step], t.steps[step], bx, by, bz, tx,
                        bdx, t.grid_y[step]);
}

/** One device thread of a fused scatter. */
template <typename W>
CARBON_HD inline void
pack_scatter_table_thread(const W* CARBON_PACK_RESTRICT stage_base,
                          W* CARBON_PACK_RESTRICT field,
                          const PackTable& t,
                          uint32_t block,
                          uint32_t tx,
                          uint32_t bdx) {
  uint32_t step = 0U;
  uint32_t bx = 0U;
  uint32_t by = 0U;
  uint32_t bz = 0U;
  if (!pack_table_decode(t, block, tx % kPackPrefixChunk, step, bx, by, bz)) {
    return;
  }
  pack_scatter_thread<W>(stage_base + t.stage_offsets[step], field, t.steps[step], bx, by, bz, tx,
                         bdx, t.grid_y[step]);
}

/** Host reference of a full gather launch (used by tests and as executable documentation). */
template <typename W>
inline auto
host_gather(const W* field,
            W* stage,
            const PackKernelArgs& a) -> void {
  const PackLaunchShape s = pack_launch_shape(a);
  for (uint32_t bz = 0; bz < s.grid_z; ++bz) {
    for (uint32_t by = 0; by < s.grid_y; ++by) {
      for (uint32_t bx = 0; bx < s.grid_x; ++bx) {
        for (uint32_t tx = 0; tx < s.block_x; ++tx) {
          pack_gather_thread<W>(field, stage, a, bx, by, bz, tx, s.block_x, s.grid_y);
        }
      }
    }
  }
}

/** Host reference of a full scatter launch. */
template <typename W>
inline auto
host_scatter(const W* stage,
             W* field,
             const PackKernelArgs& a) -> void {
  const PackLaunchShape s = pack_launch_shape(a);
  for (uint32_t bz = 0; bz < s.grid_z; ++bz) {
    for (uint32_t by = 0; by < s.grid_y; ++by) {
      for (uint32_t bx = 0; bx < s.grid_x; ++bx) {
        for (uint32_t tx = 0; tx < s.block_x; ++tx) {
          pack_scatter_thread<W>(stage, field, a, bx, by, bz, tx, s.block_x, s.grid_y);
        }
      }
    }
  }
}

/** Host reference of a fused gather launch (@p t viewing host memory). */
template <typename W>
inline auto
host_gather_table(const W* field,
                  W* stage_base,
                  const PackTable& t) -> void {
  const PackLaunchShape s = pack_table_launch_shape(t);
  for (uint32_t block = 0; block < s.grid_x; ++block) {
    for (uint32_t tx = 0; tx < s.block_x; ++tx) {
      pack_gather_table_thread<W>(field, stage_base, t, block, tx, s.block_x);
    }
  }
}

/** Host reference of a fused scatter launch (@p t viewing host memory). */
template <typename W>
inline auto
host_scatter_table(const W* stage_base,
                   W* field,
                   const PackTable& t) -> void {
  const PackLaunchShape s = pack_table_launch_shape(t);
  for (uint32_t block = 0; block < s.grid_x; ++block) {
    for (uint32_t tx = 0; tx < s.block_x; ++tx) {
      pack_scatter_table_thread<W>(stage_base, field, t, block, tx, s.block_x);
    }
  }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

enum class MsgOp : std::uint8_t {
  Send = 0,
  Recv = 1,
};

/** One kernel launch (<= kMaxPackComps components) plus where its staging region starts. */
struct PackStep {
  PackKernelArgs args{};
  size_t stage_offset{};  ///< element offset into the HaloDistributed's staging buffer
};

/** Alignment of every array in a @ref PackTableImage, and of the image size. */
inline constexpr size_t kPackTableAlign = 16U;

/**
 * The arrays of a @ref PackTable serialized into one byte image: @c steps , @c stage_offsets ,
 * @c grid_x , @c grid_y and @c block_prefix , each at a @ref kPackTableAlign aligned offset. The
 * image size is a multiple of @ref kPackTableAlign , so images concatenated back to back stay
 * aligned. Copy @c bytes anywhere @ref kPackTableAlign aligned (host or device) and @ref view it.
 */
struct PackTableImage {
  std::vector<std::byte> bytes{};
  size_t stage_offsets_at{};
  size_t grid_x_at{};
  size_t grid_y_at{};
  size_t block_prefix_at{};
  uint32_t num_steps{};
  uint32_t total_blocks{};

  /** The table, for an image whose bytes start at @p base . */
  [[nodiscard]] auto
  view(const std::byte* base) const noexcept -> PackTable {
    PackTable t;
    t.steps = reinterpret_cast<const PackKernelArgs*>(base);
    t.stage_offsets = reinterpret_cast<const size_t*>(base + stage_offsets_at);
    t.grid_x = reinterpret_cast<const uint32_t*>(base + grid_x_at);
    t.grid_y = reinterpret_cast<const uint32_t*>(base + grid_y_at);
    t.block_prefix = reinterpret_cast<const uint32_t*>(base + block_prefix_at);
    t.num_steps = num_steps;
    t.total_blocks = total_blocks;
    return t;
  }
};

namespace detail_plan {

constexpr auto
align_up(size_t n) noexcept -> size_t {
  return (n + kPackTableAlign - 1U) / kPackTableAlign * kPackTableAlign;
}

template <typename V>
auto
put_array(std::vector<std::byte>& bytes,
          size_t at,
          const std::vector<V>& values) -> void {
  std::memcpy(bytes.data() + at, values.data(), values.size() * sizeof(V));
}

}  // namespace detail_plan

/** Build the table image of @p steps (@p n >= 1, every step non-empty). */
[[nodiscard]] inline auto
make_pack_table_image(const PackStep* steps,
                      size_t n) -> PackTableImage {
  if (n == 0U || n > static_cast<size_t>(UINT32_MAX - 1U)) {
    throw std::invalid_argument("make_pack_table_image: step count out of range");
  }
  std::vector<PackKernelArgs> args(n);
  std::vector<size_t> stage_offsets(n);
  std::vector<uint32_t> grid_x(n);
  std::vector<uint32_t> grid_y(n);
  std::vector<uint32_t> block_prefix(n + 1U);
  uint64_t blocks = 0U;
  for (size_t i = 0; i < n; ++i) {
    const PackKernelArgs& a = steps[i].args;
    if (a.num_comps == 0U || a.row == 0U || a.ext_z == 0U) {
      throw std::invalid_argument("make_pack_table_image: empty step");
    }
    const PackLaunchShape s = pack_launch_shape(a);
    args[i] = a;
    stage_offsets[i] = steps[i].stage_offset;
    grid_x[i] = s.grid_x;
    grid_y[i] = s.grid_y;
    block_prefix[i] = static_cast<uint32_t>(blocks);
    blocks += static_cast<uint64_t>(s.grid_x) * s.grid_y * s.grid_z;
    if (blocks > static_cast<uint64_t>(INT32_MAX)) {
      throw std::invalid_argument("make_pack_table_image: table exceeds the 1D launch range");
    }
  }
  block_prefix[n] = static_cast<uint32_t>(blocks);

  using detail_plan::align_up;
  PackTableImage img;
  img.num_steps = static_cast<uint32_t>(n);
  img.total_blocks = static_cast<uint32_t>(blocks);
  img.stage_offsets_at = align_up(n * sizeof(PackKernelArgs));
  img.grid_x_at = align_up(img.stage_offsets_at + n * sizeof(size_t));
  img.grid_y_at = align_up(img.grid_x_at + n * sizeof(uint32_t));
  img.block_prefix_at = align_up(img.grid_y_at + n * sizeof(uint32_t));
  img.bytes.assign(align_up(img.block_prefix_at + (n + 1U) * sizeof(uint32_t)), std::byte{0});
  detail_plan::put_array(img.bytes, 0U, args);
  detail_plan::put_array(img.bytes, img.stage_offsets_at, stage_offsets);
  detail_plan::put_array(img.bytes, img.grid_x_at, grid_x);
  detail_plan::put_array(img.bytes, img.grid_y_at, grid_y);
  detail_plan::put_array(img.bytes, img.block_prefix_at, block_prefix);
  return img;
}

[[nodiscard]] inline auto
make_pack_table_image(const std::vector<PackStep>& steps) -> PackTableImage {
  return make_pack_table_image(steps.data(), steps.size());
}

/** One cross-rank message over a contiguous staging region. */
struct PackMsg {
  int peer{};
  MsgOp op{MsgOp::Send};
  size_t first{};  ///< element offset into the staging buffer
  size_t count{};  ///< elements
  /// Identical on both ends of a message: (axis, src_rank, src_side, dst_rank, dst_side) for the
  /// staged plan, (direction code from sender, src_rank, 0, dst_rank, 0) for Pencil and Brick
  /// Direct.
  std::tuple<int, int, int, int, int> key{};
};

struct AxisPackPlan {
  std::vector<PackStep> gathers{};   ///< run BEFORE the exchange (cross-rank sends + self wraps)
  std::vector<PackStep> scatters{};  ///< run AFTER the exchange (cross-rank recvs + self wraps)
  std::vector<PackMsg> msgs{};       ///< cross-rank messages (sort with @ref sort_msgs first)
  size_t stage_elems{};              ///< staging elements consumed by this plan
};

/**
 * NCCL point-to-point has no tags, so both ends of every peer pair must issue their operations in
 * the same order: all sends before all recvs per peer, ordered by a key identical on both ends.
 * With only two ranks along a periodic axis the same peer appears on both sides, which is exactly
 * why the key (not insertion order) must decide.
 */
inline auto
sort_msgs(std::vector<PackMsg>& msgs) -> void {
  std::sort(msgs.begin(), msgs.end(), [](const PackMsg& a, const PackMsg& b) {
    return std::tie(a.peer, a.op, a.key) < std::tie(b.peer, b.op, b.key);
  });
}

/**
 * x and y ranges (buffer coordinates) already filled by earlier passes, i.e. the part of a face
 * that carries valid data. Exterior halos on open axes are excluded so user boundary data there
 * is never overwritten. Both ends of a message compute the same range because their neighbours
 * along the passed axis share the other two process coordinates.
 */
struct FilledRange {
  int64_t x_lo{};
  int64_t x_hi{};
  int64_t y_lo{};
  int64_t y_hi{};
};

/** Owned face of @p side: the cells this rank sends to its neighbour on @p side. */
[[nodiscard]] inline auto
pack_face_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              const FilledRange& r) -> PackBox {
  const bool lower = side == AxisSide::Lower;
  switch (axis) {
    case Axis::X:
      return PackBox{
          lower ? g.halo.x : g.owned.x, g.halo.y, g.halo.z, g.halo.x, g.owned.y, g.owned.z};
    case Axis::Y:
      return PackBox{r.x_lo,   lower ? g.halo.y : g.owned.y, g.halo.z, r.x_hi - r.x_lo, g.halo.y,
                     g.owned.z};
    case Axis::Z:
      return PackBox{r.x_lo,          r.y_lo,          lower ? g.halo.z : g.owned.z,
                     r.x_hi - r.x_lo, r.y_hi - r.y_lo, g.halo.z};
  }
  throw std::invalid_argument("pack_face_box: unknown axis");
}

/** Halo region of @p side: the cells this rank receives from its neighbour on @p side. */
[[nodiscard]] inline auto
pack_halo_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              const FilledRange& r) -> PackBox {
  const bool lower = side == AxisSide::Lower;
  switch (axis) {
    case Axis::X:
      return PackBox{
          lower ? 0 : g.halo.x + g.owned.x, g.halo.y, g.halo.z, g.halo.x, g.owned.y, g.owned.z};
    case Axis::Y:
      return PackBox{
          r.x_lo, lower ? 0 : g.halo.y + g.owned.y, g.halo.z, r.x_hi - r.x_lo, g.halo.y, g.owned.z};
    case Axis::Z:
      return PackBox{r.x_lo,          r.y_lo,          lower ? 0 : g.halo.z + g.owned.z,
                     r.x_hi - r.x_lo, r.y_hi - r.y_lo, g.halo.z};
  }
  throw std::invalid_argument("pack_halo_box: unknown axis");
}

/** Convenience overloads keeping the X/Y-only call shape (y range = whole haloed y). */
[[nodiscard]] inline auto
pack_face_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              int64_t x_lo,
              int64_t x_hi) -> PackBox {
  return pack_face_box(g, axis, side, FilledRange{x_lo, x_hi, 0, g.haloed.y});
}

[[nodiscard]] inline auto
pack_halo_box(const AxisGeometry& g,
              Axis axis,
              AxisSide side,
              int64_t x_lo,
              int64_t x_hi) -> PackBox {
  return pack_halo_box(g, axis, side, FilledRange{x_lo, x_hi, 0, g.haloed.y});
}

namespace detail_plan {

/** Append one launch per chunk of <= kMaxPackComps components for @p box / @p comps. */
inline auto
append_steps(std::vector<PackStep>& out,
             const PackBox& box,
             const std::vector<int>& comps,
             const AxisGeometry& g,
             int64_t comp_stride,
             size_t region_offset) -> void {
  if (box.volume() == 0U || comps.empty()) {
    return;
  }
  if (box.ext_x * box.ext_y > static_cast<int64_t>(UINT32_MAX) ||
      box.ext_z > static_cast<int64_t>(UINT32_MAX)) {
    throw std::invalid_argument("halo pack: face extent exceeds 32-bit launch range");
  }
  const int64_t per_comp = static_cast<int64_t>(box.volume());
  for (size_t first = 0; first < comps.size(); first += kMaxPackComps) {
    const size_t n = std::min<size_t>(kMaxPackComps, comps.size() - first);
    PackStep step;
    PackKernelArgs& a = step.args;
    a.lo_x = box.lo_x;
    a.lo_y = box.lo_y;
    a.lo_z = box.lo_z;
    a.haloed_x = g.haloed.x;
    a.pitch = g.pitch;
    a.comp_stride = comp_stride;
    a.stage_comp_stride = per_comp;
    a.ext_x = static_cast<uint32_t>(box.ext_x);
    a.ext_y = static_cast<uint32_t>(box.ext_y);
    a.ext_z = static_cast<uint32_t>(box.ext_z);
    a.row = a.ext_x * a.ext_y;
    a.num_comps = static_cast<uint32_t>(n);
    for (size_t k = 0; k < n; ++k) {
      a.comps[k] = comps[first + k];
    }
    step.stage_offset = region_offset + first * static_cast<size_t>(per_comp);
    out.push_back(step);
  }
}

}  // namespace detail_plan

/**
 * Build the packed exchange plan for one X or Y pass of one rank.
 *
 * @param stage_base element offset at which this plan's staging regions start (so several passes
 *                   can share one staging buffer)
 * @param active_send / active_recv  (component, axis, side) -> bool; same predicates the
 *                   run-based builder uses, so lattice direction filtering carries over.
 *
 * Semantics per side, identical to the run-based builder:
 *  - cross-rank: send OUR owned face at `side` to the neighbour on `side`; receive INTO OUR halo at
 *    `side` from that neighbour's opposite face.
 *  - self wrap (periodic axis owned by a single rank): halo at `side` <- OUR owned face on the
 *    opposite side, staged through the same buffer (no message).
 */
template <typename SendActive,
          typename RecvActive>
[[nodiscard]] inline auto
build_axis_pack_plan(const ProcessGrid& pg,
                     BoundaryModeDistributed boundary,
                     int32_3d coords,
                     int32_t local_rank,
                     const AxisGeometry& g,
                     Axis axis,
                     int components,
                     int64_t comp_stride,
                     SendActive&& active_send,
                     RecvActive&& active_recv,
                     size_t stage_base) -> AxisPackPlan {
  AxisPackPlan plan;
  const int64_t owned = axis == Axis::X ? g.owned.x : (axis == Axis::Y ? g.owned.y : g.owned.z);
  const int64_t halo = axis == Axis::X ? g.halo.x : (axis == Axis::Y ? g.halo.y : g.halo.z);
  if (halo <= 0) {
    return plan;
  }
  if (owned < halo) {
    throw std::invalid_argument("halo pack: owned extent is smaller than the halo");
  }

  // Ranges already filled when this pass starts (see file comment). Neighbours along this axis
  // share our other two coordinates, so both ends of a message compute the same ranges.
  FilledRange fill{0, g.haloed.x, 0, g.haloed.y};
  if (axis != Axis::X && g.halo.x > 0) {
    const bool has_lo =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::X, boundary, AxisSide::Lower)
            .present;
    const bool has_hi =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::X, boundary, AxisSide::Upper)
            .present;
    fill.x_lo = has_lo ? 0 : g.halo.x;
    fill.x_hi = g.haloed.x - (has_hi ? 0 : g.halo.x);
  }
  if (axis == Axis::Z && g.halo.y > 0) {
    const bool has_lo =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::Y, boundary, AxisSide::Lower)
            .present;
    const bool has_hi =
        halo_distributed_build::axis_neighbour(pg, coords, Axis::Y, boundary, AxisSide::Upper)
            .present;
    fill.y_lo = has_lo ? 0 : g.halo.y;
    fill.y_hi = g.haloed.y - (has_hi ? 0 : g.halo.y);
  }

  size_t cursor = stage_base;
  const int axis_id = static_cast<int>(axis);

  for (const AxisSide side : {AxisSide::Lower, AxisSide::Upper}) {
    const auto nb = halo_distributed_build::axis_neighbour(pg, coords, axis, boundary, side);
    if (!nb.present) {
      continue;
    }
    const AxisSide opposite = side == AxisSide::Lower ? AxisSide::Upper : AxisSide::Lower;
    const int local_side = static_cast<int>(side);
    const int peer_side = 1 - local_side;

    std::vector<int> send_comps;
    std::vector<int> recv_comps;
    for (int c = 0; c < components; ++c) {
      if (active_send(c, axis, side)) {
        send_comps.push_back(c);
      }
      if (active_recv(c, axis, side)) {
        recv_comps.push_back(c);
      }
    }

    const PackBox halo_box = pack_halo_box(g, axis, side, fill);

    if (nb.rank == local_rank) {
      // Self wrap: gather our `opposite` owned face, scatter into our `side` halo.
      const PackBox src_box = pack_face_box(g, axis, opposite, fill);
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, src_box, recv_comps, g, comp_stride, region);
      detail_plan::append_steps(plan.scatters, halo_box, recv_comps, g, comp_stride, region);
      cursor += recv_comps.size() * halo_box.volume();
      continue;
    }

    // Cross-rank: one send region + one recv region per side.
    const PackBox send_box = pack_face_box(g, axis, side, fill);
    if (!send_comps.empty() && send_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, send_box, send_comps, g, comp_stride, region);
      const size_t count = send_comps.size() * send_box.volume();
      plan.msgs.push_back({nb.rank,
                           MsgOp::Send,
                           region,
                           count,
                           {axis_id, static_cast<int>(local_rank), local_side,
                            static_cast<int>(nb.rank), peer_side}});
      cursor += count;
    }
    if (!recv_comps.empty() && halo_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.scatters, halo_box, recv_comps, g, comp_stride, region);
      const size_t count = recv_comps.size() * halo_box.volume();
      plan.msgs.push_back({nb.rank,
                           MsgOp::Recv,
                           region,
                           count,
                           {axis_id, static_cast<int>(nb.rank), peer_side,
                            static_cast<int>(local_rank), local_side}});
      cursor += count;
    }
  }

  plan.stage_elems = cursor - stage_base;
  return plan;
}

// ---------------------------------------------------------------------------
// Pencil Direct plan (HaloExchangeMode::Direct)
// ---------------------------------------------------------------------------

/** Direction to a neighbour in process-grid @c (y, z) coordinates; each component in -1..1. */
struct PencilDirectDir {
  int32_t dy{};
  int32_t dz{};
};

/** The four faces and four YZ edges of a Pencil span. */
inline constexpr PencilDirectDir kPencilDirectDirs[8] = {
    {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
};

/** Distinct code per direction, used in message keys. */
[[nodiscard]] constexpr auto
pencil_direct_code(PencilDirectDir d) noexcept -> int {
  return (d.dy + 1) * 3 + (d.dz + 1);
}

/**
 * Rank and coordinates of the neighbour at @p coords + @p d , wrapping periodic axes; absent past an
 * open edge of the process grid (on either axis of an edge direction).
 */
[[nodiscard]] inline auto
pencil_direct_neighbour(const ProcessGrid& pg,
                        int32_3d coords,
                        PencilDirectDir d,
                        BoundaryModeDistributed boundary) -> halo_distributed_build::AxisNeighbour {
  using halo_distributed_build::axis_neighbour;
  halo_distributed_build::AxisNeighbour nb{true, true, coords, pg.rank_of(coords)};
  if (d.dy != 0) {
    nb = axis_neighbour(pg, nb.coords, Axis::Y, boundary,
                        d.dy < 0 ? AxisSide::Lower : AxisSide::Upper);
    if (!nb.present) {
      return {};
    }
  }
  if (d.dz != 0) {
    nb = axis_neighbour(pg, nb.coords, Axis::Z, boundary,
                        d.dz < 0 ? AxisSide::Lower : AxisSide::Upper);
    if (!nb.present) {
      return {};
    }
  }
  nb.self = nb.coords.y == coords.y && nb.coords.z == coords.z;
  return nb;
}

namespace detail_plan {

/** Owned cells along one axis that the neighbour at step @p k needs (@p k in -1..1). */
constexpr auto
direct_send_range(int32_t k,
                  int64_t owned,
                  int64_t halo,
                  int64_t& lo,
                  int64_t& ext) noexcept -> void {
  lo = k > 0 ? owned : halo;
  ext = k == 0 ? owned : halo;
}

/** Halo cells along one axis filled from the neighbour at step @p k . */
constexpr auto
direct_recv_range(int32_t k,
                  int64_t owned,
                  int64_t halo,
                  int64_t& lo,
                  int64_t& ext) noexcept -> void {
  lo = k < 0 ? 0 : (k > 0 ? halo + owned : halo);
  ext = k == 0 ? owned : halo;
}

}  // namespace detail_plan

/** Owned box this rank sends to the neighbour at @p d (full x; pencil spans have no x halo). */
[[nodiscard]] inline auto
pencil_direct_send_box(const AxisGeometry& g,
                       PencilDirectDir d) -> PackBox {
  PackBox b{0, 0, 0, g.haloed.x, 0, 0};
  detail_plan::direct_send_range(d.dy, g.owned.y, g.halo.y, b.lo_y, b.ext_y);
  detail_plan::direct_send_range(d.dz, g.owned.z, g.halo.z, b.lo_z, b.ext_z);
  return b;
}

/** Halo box this rank fills from the neighbour at @p d . */
[[nodiscard]] inline auto
pencil_direct_recv_box(const AxisGeometry& g,
                       PencilDirectDir d) -> PackBox {
  PackBox b{0, 0, 0, g.haloed.x, 0, 0};
  detail_plan::direct_recv_range(d.dy, g.owned.y, g.halo.y, b.lo_y, b.ext_y);
  detail_plan::direct_recv_range(d.dz, g.owned.z, g.halo.z, b.lo_z, b.ext_z);
  return b;
}

/**
 * Build the single-pass packed plan of a Pencil span (@c halo.x == 0, @c halo.y and @c halo.z > 0)
 * for one rank: faces and YZ edges are exchanged directly with all eight neighbours, so no halo
 * cell depends on another exchange and the whole update is one gather, one message group and one
 * scatter.
 *
 * @param send_active (component, dy, dz) -> bool: component goes to the neighbour at (dy, dz).
 * @param recv_active (component, dy, dz) -> bool: component comes from the neighbour at (dy, dz).
 * Both ends of a message evaluate the same set because recv_active(c, -d) on the receiver must
 * equal send_active(c, d) on the sender (true for the stencil semantics HaloDistributed builds).
 *
 * Per direction: cross-rank sends this rank's owned box and receives into its halo box; a
 * self-wrap (periodic axis owned by this rank alone) copies the owned box of the opposite
 * direction into the halo box through the staging buffer, with no message. Absent neighbours
 * (open axes) leave the corresponding halo cells untouched, matching the staged path.
 */
template <typename SendActive,
          typename RecvActive>
[[nodiscard]] inline auto
build_pencil_direct_pack_plan(const ProcessGrid& pg,
                              BoundaryModeDistributed boundary,
                              int32_3d coords,
                              int32_t local_rank,
                              const AxisGeometry& g,
                              int components,
                              int64_t comp_stride,
                              SendActive&& send_active,
                              RecvActive&& recv_active,
                              size_t stage_base) -> AxisPackPlan {
  AxisPackPlan plan;
  if (g.halo.x != 0 || g.halo.y <= 0 || g.halo.z <= 0) {
    throw std::invalid_argument(
        "halo pack: Pencil Direct needs halo.x == 0 and halo.y, halo.z > 0");
  }
  if (g.owned.y < g.halo.y || g.owned.z < g.halo.z) {
    throw std::invalid_argument("halo pack: owned extent is smaller than the halo");
  }

  size_t cursor = stage_base;
  for (const PencilDirectDir d : kPencilDirectDirs) {
    const auto nb = pencil_direct_neighbour(pg, coords, d, boundary);
    if (!nb.present) {
      continue;
    }
    const PencilDirectDir opposite{-d.dy, -d.dz};

    std::vector<int> send_comps;
    std::vector<int> recv_comps;
    for (int c = 0; c < components; ++c) {
      if (send_active(c, d.dy, d.dz)) {
        send_comps.push_back(c);
      }
      if (recv_active(c, d.dy, d.dz)) {
        recv_comps.push_back(c);
      }
    }

    const PackBox recv_box = pencil_direct_recv_box(g, d);

    if (nb.rank == local_rank) {
      const PackBox src_box = pencil_direct_send_box(g, opposite);
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, src_box, recv_comps, g, comp_stride, region);
      detail_plan::append_steps(plan.scatters, recv_box, recv_comps, g, comp_stride, region);
      cursor += recv_comps.size() * recv_box.volume();
      continue;
    }

    const PackBox send_box = pencil_direct_send_box(g, d);
    if (!send_comps.empty() && send_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.gathers, send_box, send_comps, g, comp_stride, region);
      const size_t count = send_comps.size() * send_box.volume();
      plan.msgs.push_back(
          {nb.rank,
           MsgOp::Send,
           region,
           count,
           {pencil_direct_code(d), static_cast<int>(local_rank), 0, static_cast<int>(nb.rank), 0}});
      cursor += count;
    }
    if (!recv_comps.empty() && recv_box.volume() > 0U) {
      const size_t region = cursor;
      detail_plan::append_steps(plan.scatters, recv_box, recv_comps, g, comp_stride, region);
      const size_t count = recv_comps.size() * recv_box.volume();
      plan.msgs.push_back({nb.rank,
                           MsgOp::Recv,
                           region,
                           count,
                           {pencil_direct_code(opposite), static_cast<int>(nb.rank), 0,
                            static_cast<int>(local_rank), 0}});
      cursor += count;
    }
  }

  plan.stage_elems = cursor - stage_base;
  return plan;
}

// ---------------------------------------------------------------------------
// Brick Direct plan (HaloExchangeMode::Direct)
// ---------------------------------------------------------------------------

/** Direction to a neighbour in process-grid @c (x, y, z) coordinates; each component in -1..1. */
struct DirectDir {
  int32_t dx{};
  int32_t dy{};
  int32_t dz{};
};

/** The six faces, twelve edges and eight corners of a Brick span. */
inline constexpr DirectDir kBrickDirectDirs[26] = {
    {1, 0, 0},   {-1, 0, 0}, {0, 1, 0},   {0, -1, 0},  {0, 0, 1},    {0, 0, -1}, {1, 1, 0},
    {1, -1, 0},  {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1},   {1, 0, -1},   {-1, 0, 1}, {-1, 0, -1},
    {0, 1, 1},   {0, 1, -1}, {0, -1, 1},  {0, -1, -1}, {1, 1, 1},    {1, 1, -1}, {1, -1, 1},
    {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1},
};

/** Distinct code per direction, used in message keys. */
[[nodiscard]] constexpr auto
direct_code(DirectDir d) noexcept -> int {
  return (d.dx + 1) * 9 + (d.dy + 1) * 3 + (d.dz + 1);
}

/**
 * Rank and coordinates of the neighbour at @p coords + @p d , wrapping periodic axes; absent past an
 * open edge of the process grid on any axis of the direction.
 */
[[nodiscard]] inline auto
direct_neighbour(const ProcessGrid& pg,
                 int32_3d coords,
                 DirectDir d,
                 BoundaryModeDistributed boundary) -> halo_distributed_build::AxisNeighbour {
  using halo_distributed_build::axis_neighbour;
  halo_distributed_build::AxisNeighbour nb{true, true, coords, pg.rank_of(coords)};
  const int32_t steps[3] = {d.dx, d.dy, d.dz};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
  for (int i = 0; i < 3; ++i) {
    if (steps[i] == 0) {
      continue;
    }
    nb = axis_neighbour(pg, nb.coords, axes[i], boundary,
                        steps[i] < 0 ? AxisSide::Lower : AxisSide::Upper);
    if (!nb.present) {
      return {};
    }
  }
  nb.self = nb.coords.x == coords.x && nb.coords.y == coords.y && nb.coords.z == coords.z;
  return nb;
}

/** Owned box this rank sends to the neighbour at @p d . */
[[nodiscard]] inline auto
brick_direct_send_box(const AxisGeometry& g,
                      DirectDir d) -> PackBox {
  PackBox b;
  detail_plan::direct_send_range(d.dx, g.owned.x, g.halo.x, b.lo_x, b.ext_x);
  detail_plan::direct_send_range(d.dy, g.owned.y, g.halo.y, b.lo_y, b.ext_y);
  detail_plan::direct_send_range(d.dz, g.owned.z, g.halo.z, b.lo_z, b.ext_z);
  return b;
}

/** Halo box this rank fills from the neighbour at @p d . */
[[nodiscard]] inline auto
brick_direct_recv_box(const AxisGeometry& g,
                      DirectDir d) -> PackBox {
  PackBox b;
  detail_plan::direct_recv_range(d.dx, g.owned.x, g.halo.x, b.lo_x, b.ext_x);
  detail_plan::direct_recv_range(d.dy, g.owned.y, g.halo.y, b.lo_y, b.ext_y);
  detail_plan::direct_recv_range(d.dz, g.owned.z, g.halo.z, b.lo_z, b.ext_z);
  return b;
}

/**
 * True for a box too narrow in x and y to fill a @c kPackBlockX row but long in z (the edges along
 * z of a brick). Such boxes are packed with y and z swapped so the long axis runs along the row.
 * Depends on the extents only, so both ends of a message (and both halves of a self wrap) agree on
 * the staging layout.
 */
[[nodiscard]] constexpr auto
pack_swap_yz(const PackBox& b) noexcept -> bool {
  return b.ext_x * b.ext_y < static_cast<int64_t>(kPackBlockX) && b.ext_z > b.ext_y;
}

namespace detail_plan {

/** @ref append_steps , packing @ref pack_swap_yz boxes with the y and z roles exchanged. */
inline auto
append_oriented_steps(std::vector<PackStep>& out,
                      const PackBox& box,
                      const std::vector<int>& comps,
                      const AxisGeometry& g,
                      int64_t comp_stride,
                      size_t region_offset) -> void {
  if (!pack_swap_yz(box)) {
    append_steps(out, box, comps, g, comp_stride, region_offset);
    return;
  }
  const PackBox swapped{box.lo_x, box.lo_z, box.lo_y, box.ext_x, box.ext_z, box.ext_y};
  const size_t first = out.size();
  append_steps(out, swapped, comps, g, comp_stride, region_offset);
  for (size_t i = first; i < out.size(); ++i) {
    std::swap(out[i].args.haloed_x, out[i].args.pitch);
  }
}

}  // namespace detail_plan

/**
 * Build the single-pass packed plan of a Brick span (@c halo.x > 0) for one rank: faces, edges and
 * corners are exchanged directly with all 26 neighbours, so no halo cell depends on another
 * exchange and the whole update is one gather, one message group and one scatter. Same contract
 * as @ref build_pencil_direct_pack_plan , with predicates taking (component, dx, dy, dz).
 *
 * Directions whose box is empty (an axis without a halo) are skipped.
 */
template <typename SendActive,
          typename RecvActive>
[[nodiscard]] inline auto
build_brick_direct_pack_plan(const ProcessGrid& pg,
                             BoundaryModeDistributed boundary,
                             int32_3d coords,
                             int32_t local_rank,
                             const AxisGeometry& g,
                             int components,
                             int64_t comp_stride,
                             SendActive&& send_active,
                             RecvActive&& recv_active,
                             size_t stage_base) -> AxisPackPlan {
  AxisPackPlan plan;
  if (g.halo.x <= 0) {
    throw std::invalid_argument("halo pack: Brick Direct needs halo.x > 0");
  }
  if (g.owned.x < g.halo.x || g.owned.y < g.halo.y || g.owned.z < g.halo.z) {
    throw std::invalid_argument("halo pack: owned extent is smaller than the halo");
  }

  size_t cursor = stage_base;
  for (const DirectDir d : kBrickDirectDirs) {
    const PackBox recv_box = brick_direct_recv_box(g, d);
    if (recv_box.volume() == 0U) {
      continue;
    }
    const auto nb = direct_neighbour(pg, coords, d, boundary);
    if (!nb.present) {
      continue;
    }
    const DirectDir opposite{-d.dx, -d.dy, -d.dz};

    std::vector<int> send_comps;
    std::vector<int> recv_comps;
    for (int c = 0; c < components; ++c) {
      if (send_active(c, d.dx, d.dy, d.dz)) {
        send_comps.push_back(c);
      }
      if (recv_active(c, d.dx, d.dy, d.dz)) {
        recv_comps.push_back(c);
      }
    }

    if (nb.rank == local_rank) {
      const PackBox src_box = brick_direct_send_box(g, opposite);
      const size_t region = cursor;
      detail_plan::append_oriented_steps(plan.gathers, src_box, recv_comps, g, comp_stride, region);
      detail_plan::append_oriented_steps(plan.scatters, recv_box, recv_comps, g, comp_stride,
                                         region);
      cursor += recv_comps.size() * recv_box.volume();
      continue;
    }

    const PackBox send_box = brick_direct_send_box(g, d);
    if (!send_comps.empty()) {
      const size_t region = cursor;
      detail_plan::append_oriented_steps(plan.gathers, send_box, send_comps, g, comp_stride,
                                         region);
      const size_t count = send_comps.size() * send_box.volume();
      plan.msgs.push_back(
          {nb.rank,
           MsgOp::Send,
           region,
           count,
           {direct_code(d), static_cast<int>(local_rank), 0, static_cast<int>(nb.rank), 0}});
      cursor += count;
    }
    if (!recv_comps.empty()) {
      const size_t region = cursor;
      detail_plan::append_oriented_steps(plan.scatters, recv_box, recv_comps, g, comp_stride,
                                         region);
      const size_t count = recv_comps.size() * recv_box.volume();
      plan.msgs.push_back(
          {nb.rank,
           MsgOp::Recv,
           region,
           count,
           {direct_code(opposite), static_cast<int>(nb.rank), 0, static_cast<int>(local_rank), 0}});
      cursor += count;
    }
  }

  plan.stage_elems = cursor - stage_base;
  return plan;
}

}  // namespace carbon::detail::halo_pack
