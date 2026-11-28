// Packed halo gather/scatter kernels (CUDA and HIP backends). See include/carbon/detail/halo_pack.h.
//
// Each __global__ is a one-line forwarder to the host+device functions in halo_pack_plan.h, which
// tests/test_halo_pack_plan.cpp and tests/test_dense_distributed_host.cpp exercise on the host with
// the same arguments. Keep the kernels trivial: any logic added here is NOT covered by that test.

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/halo_pack.h>

#include <cstddef>
#include <cstdint>

namespace carbon::detail::halo_pack {
namespace {

struct alignas(16) Word16 {
  std::uint64_t lo;
  std::uint64_t hi;
};

template <typename W>
__global__ void
gather_step_kernel(const W* __restrict__ field,
                   W* __restrict__ stage,
                   PackKernelArgs a) {
  pack_gather_thread<W>(field, stage, a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
                        blockDim.x, gridDim.y);
}

template <typename W>
__global__ void
scatter_step_kernel(const W* __restrict__ stage,
                    W* __restrict__ field,
                    PackKernelArgs a) {
  pack_scatter_thread<W>(stage, field, a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
                         blockDim.x, gridDim.y);
}

template <typename W>
__global__ void
gather_table_kernel(const W* __restrict__ field,
                    W* __restrict__ stage_base,
                    PackTable t) {
  pack_gather_table_thread<W>(field, stage_base, t, blockIdx.x, threadIdx.x, blockDim.x);
}

template <typename W>
__global__ void
scatter_table_kernel(const W* __restrict__ stage_base,
                     W* __restrict__ field,
                     PackTable t) {
  pack_scatter_table_thread<W>(stage_base, field, t, blockIdx.x, threadIdx.x, blockDim.x);
}

template <typename W>
auto
gather_step_as(const Gpu& gpu,
               gpu_stream_t stream,
               const void* field,
               void* stage,
               const PackKernelArgs& a) -> gpu_error_t {
  if (a.num_comps == 0U || a.row == 0U || a.ext_z == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_launch_shape(a);
  return gpu.template typed_launch<&gather_step_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(field), static_cast<W*>(stage), a);
}

template <typename W>
auto
scatter_step_as(const Gpu& gpu,
                gpu_stream_t stream,
                const void* stage,
                void* field,
                const PackKernelArgs& a) -> gpu_error_t {
  if (a.num_comps == 0U || a.row == 0U || a.ext_z == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_launch_shape(a);
  return gpu.template typed_launch<&scatter_step_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(stage), static_cast<W*>(field), a);
}

template <typename W>
auto
gather_table_as(const Gpu& gpu,
                gpu_stream_t stream,
                const void* field,
                void* stage_base,
                const PackTable& t) -> gpu_error_t {
  if (t.num_steps == 0U || t.total_blocks == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_table_launch_shape(t);
  return gpu.template typed_launch<&gather_table_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(field), static_cast<W*>(stage_base), t);
}

template <typename W>
auto
scatter_table_as(const Gpu& gpu,
                 gpu_stream_t stream,
                 const void* stage_base,
                 void* field,
                 const PackTable& t) -> gpu_error_t {
  if (t.num_steps == 0U || t.total_blocks == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_table_launch_shape(t);
  return gpu.template typed_launch<&scatter_table_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(stage_base), static_cast<W*>(field), t);
}

/** Calls @p f with a value of the word type matching @p elem_bytes (unsupported -> invalid). */
template <typename F>
auto
dispatch_width(std::size_t elem_bytes,
               F&& f) -> gpu_error_t {
  switch (elem_bytes) {
    case 1U:
      return f(std::uint8_t{});
    case 2U:
      return f(std::uint16_t{});
    case 4U:
      return f(std::uint32_t{});
    case 8U:
      return f(std::uint64_t{});
    case 16U:
      return f(Word16{});
    default:
      return gpu_error_invalid_value;
  }
}

}  // namespace

auto
launch_gather(const Gpu& gpu,
              gpu_stream_t stream,
              std::size_t elem_bytes,
              const void* field_base,
              void* stage_region,
              const PackKernelArgs& a) -> gpu_error_t {
  return dispatch_width(elem_bytes, [&](auto w) {
    return gather_step_as<decltype(w)>(gpu, stream, field_base, stage_region, a);
  });
}

auto
launch_scatter(const Gpu& gpu,
               gpu_stream_t stream,
               std::size_t elem_bytes,
               const void* stage_region,
               void* field_base,
               const PackKernelArgs& a) -> gpu_error_t {
  return dispatch_width(elem_bytes, [&](auto w) {
    return scatter_step_as<decltype(w)>(gpu, stream, stage_region, field_base, a);
  });
}

auto
launch_gather_table(const Gpu& gpu,
                    gpu_stream_t stream,
                    std::size_t elem_bytes,
                    const void* field_base,
                    void* stage_base,
                    const PackTable& table) -> gpu_error_t {
  return dispatch_width(elem_bytes, [&](auto w) {
    return gather_table_as<decltype(w)>(gpu, stream, field_base, stage_base, table);
  });
}

auto
launch_scatter_table(const Gpu& gpu,
                     gpu_stream_t stream,
                     std::size_t elem_bytes,
                     const void* stage_base,
                     void* field_base,
                     const PackTable& table) -> gpu_error_t {
  return dispatch_width(elem_bytes, [&](auto w) {
    return scatter_table_as<decltype(w)>(gpu, stream, stage_base, field_base, table);
  });
}

}  // namespace carbon::detail::halo_pack
