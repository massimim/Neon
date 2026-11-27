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
gather_batch_kernel(const W* __restrict__ field,
                    W* __restrict__ stage_base,
                    PackBatch b) {
  pack_gather_batch_thread<W>(field, stage_base, b, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
                              blockDim.x, gridDim.y);
}

template <typename W>
__global__ void
scatter_batch_kernel(const W* __restrict__ stage_base,
                     W* __restrict__ field,
                     PackBatch b) {
  pack_scatter_batch_thread<W>(stage_base, field, b, blockIdx.x, blockIdx.y, blockIdx.z,
                               threadIdx.x, blockDim.x, gridDim.y);
}

template <typename W>
auto
gather_batch_as(const Gpu& gpu,
                gpu_stream_t stream,
                const void* field,
                void* stage_base,
                const PackBatch& b) -> gpu_error_t {
  if (b.num_steps == 0U || b.total_z == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_batch_launch_shape(b);
  return gpu.template typed_launch<&gather_batch_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(field), static_cast<W*>(stage_base), b);
}

template <typename W>
auto
scatter_batch_as(const Gpu& gpu,
                 gpu_stream_t stream,
                 const void* stage_base,
                 void* field,
                 const PackBatch& b) -> gpu_error_t {
  if (b.num_steps == 0U || b.total_z == 0U) {
    return gpu_success;
  }
  const PackLaunchShape s = pack_batch_launch_shape(b);
  return gpu.template typed_launch<&scatter_batch_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(stage_base), static_cast<W*>(field), b);
}

auto
dispatch_gather_batch(const Gpu& gpu,
                      gpu_stream_t stream,
                      std::size_t elem_bytes,
                      const void* field_base,
                      void* stage_base,
                      const PackBatch& batch) -> gpu_error_t {
  switch (elem_bytes) {
    case 1U:
      return gather_batch_as<std::uint8_t>(gpu, stream, field_base, stage_base, batch);
    case 2U:
      return gather_batch_as<std::uint16_t>(gpu, stream, field_base, stage_base, batch);
    case 4U:
      return gather_batch_as<std::uint32_t>(gpu, stream, field_base, stage_base, batch);
    case 8U:
      return gather_batch_as<std::uint64_t>(gpu, stream, field_base, stage_base, batch);
    case 16U:
      return gather_batch_as<Word16>(gpu, stream, field_base, stage_base, batch);
    default:
      return gpu_error_invalid_value;
  }
}

auto
dispatch_scatter_batch(const Gpu& gpu,
                       gpu_stream_t stream,
                       std::size_t elem_bytes,
                       const void* stage_base,
                       void* field_base,
                       const PackBatch& batch) -> gpu_error_t {
  switch (elem_bytes) {
    case 1U:
      return scatter_batch_as<std::uint8_t>(gpu, stream, stage_base, field_base, batch);
    case 2U:
      return scatter_batch_as<std::uint16_t>(gpu, stream, stage_base, field_base, batch);
    case 4U:
      return scatter_batch_as<std::uint32_t>(gpu, stream, stage_base, field_base, batch);
    case 8U:
      return scatter_batch_as<std::uint64_t>(gpu, stream, stage_base, field_base, batch);
    case 16U:
      return scatter_batch_as<Word16>(gpu, stream, stage_base, field_base, batch);
    default:
      return gpu_error_invalid_value;
  }
}

}  // namespace

auto
launch_gather_batch(const Gpu& gpu,
                    gpu_stream_t stream,
                    std::size_t elem_bytes,
                    const void* field_base,
                    void* stage_base,
                    const PackBatch& batch) -> gpu_error_t {
  return dispatch_gather_batch(gpu, stream, elem_bytes, field_base, stage_base, batch);
}

auto
launch_scatter_batch(const Gpu& gpu,
                     gpu_stream_t stream,
                     std::size_t elem_bytes,
                     const void* stage_base,
                     void* field_base,
                     const PackBatch& batch) -> gpu_error_t {
  return dispatch_scatter_batch(gpu, stream, elem_bytes, stage_base, field_base, batch);
}

auto
launch_gather(const Gpu& gpu,
              gpu_stream_t stream,
              std::size_t elem_bytes,
              const void* field_base,
              void* stage_region,
              const PackKernelArgs& a) -> gpu_error_t {
  PackStep step;
  step.args = a;
  step.stage_offset = 0U;
  return launch_gather_batch(gpu, stream, elem_bytes, field_base, stage_region,
                             make_pack_batch(&step, 1U));
}

auto
launch_scatter(const Gpu& gpu,
               gpu_stream_t stream,
               std::size_t elem_bytes,
               const void* stage_region,
               void* field_base,
               const PackKernelArgs& a) -> gpu_error_t {
  PackStep step;
  step.args = a;
  step.stage_offset = 0U;
  return launch_scatter_batch(gpu, stream, elem_bytes, stage_region, field_base,
                              make_pack_batch(&step, 1U));
}

}  // namespace carbon::detail::halo_pack
