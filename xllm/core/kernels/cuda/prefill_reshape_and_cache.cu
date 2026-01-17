/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cstdint>
#include <type_traits>

#include "cuda_ops_api.h"
#include "utils.h"

using at::device_of;

namespace {

// Vectorized type traits for 128-bit vectorization
template <typename scalar_t>
struct VecType;
template <>
struct VecType<c10::Half> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int vec_width = 8;
};
template <>
struct VecType<c10::BFloat16> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int vec_width = 8;
};
template <>
struct VecType<float> {
  using type = float4;  // 4 elements * 4 bytes = 16 bytes
  static constexpr int vec_width = 4;
};

// Vectorized kernel with 128-bit vectorization for contiguous destination
// Uses scalar load from strided source, vectorized store to contiguous
// destination
template <typename scalar_t>
__global__ void prefill_reshape_and_cache_kernel_vec(
    const scalar_t* __restrict__ proj_k,    // [shared_len, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,    // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_k_cache,  // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_v_cache,  // [shared_len, kv_heads, head_dim]
    const int64_t shared_len,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t k_stride0,    // proj_k.stride(0)
    const int64_t k_stride1,    // proj_k.stride(1)
    const int64_t v_stride0,    // proj_v.stride(0)
    const int64_t v_stride1) {  // proj_v.stride(1)
  using VecTypeT = typename VecType<scalar_t>::type;
  constexpr int VEC_WIDTH = VecType<scalar_t>::vec_width;

  const int64_t token_idx = static_cast<int64_t>(blockIdx.y);
  if (token_idx >= shared_len) {
    return;
  }

  const int64_t total_elements = kv_heads * head_dim;
  const int64_t vec_elements = (total_elements / VEC_WIDTH) * VEC_WIDTH;
  const int64_t base_dst_idx = token_idx * kv_heads * head_dim;

  // Vectorized path: process aligned elements
  for (int64_t i = threadIdx.x * VEC_WIDTH; i < vec_elements;
       i += blockDim.x * VEC_WIDTH) {
    const int64_t head_idx = i / head_dim;
    const int64_t dim_idx = i % head_dim;

    // Scalar load from strided source into vector register
    VecTypeT k_vec, v_vec;
    scalar_t* k_vec_ptr = reinterpret_cast<scalar_t*>(&k_vec);
    scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);

    // Since head_dim % VEC_WIDTH == 0 (checked before kernel launch),
    // all VEC_WIDTH elements are within the same head
#pragma unroll
    for (int j = 0; j < VEC_WIDTH; ++j) {
      const int64_t current_dim = dim_idx + j;
      const int64_t k_src_idx =
          token_idx * k_stride0 + head_idx * k_stride1 + current_dim;
      const int64_t v_src_idx =
          token_idx * v_stride0 + head_idx * v_stride1 + current_dim;
      k_vec_ptr[j] = proj_k[k_src_idx];
      v_vec_ptr[j] = proj_v[v_src_idx];
    }

    // Vectorized store to contiguous destination
    const int64_t dst_idx = base_dst_idx + i;
    *reinterpret_cast<VecTypeT*>(&shared_k_cache[dst_idx]) = k_vec;
    *reinterpret_cast<VecTypeT*>(&shared_v_cache[dst_idx]) = v_vec;
  }

  // Scalar fallback: process remaining elements
  for (int64_t i = vec_elements + threadIdx.x; i < total_elements;
       i += blockDim.x) {
    const int64_t head_idx = i / head_dim;
    const int64_t dim_idx = i % head_dim;

    // Source index (considering stride for non-contiguous input)
    const int64_t k_src_idx =
        token_idx * k_stride0 + head_idx * k_stride1 + dim_idx;
    const int64_t v_src_idx =
        token_idx * v_stride0 + head_idx * v_stride1 + dim_idx;

    // Destination index (contiguous output)
    const int64_t dst_idx = base_dst_idx + i;

    shared_k_cache[dst_idx] = proj_k[k_src_idx];
    shared_v_cache[dst_idx] = proj_v[v_src_idx];
  }
}

// Scalar fallback kernel (when vectorization is not possible)
template <typename scalar_t>
__global__ void prefill_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,    // [shared_len, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,    // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_k_cache,  // [shared_len, kv_heads, head_dim]
    scalar_t* __restrict__ shared_v_cache,  // [shared_len, kv_heads, head_dim]
    const int64_t shared_len,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t k_stride0,    // proj_k.stride(0)
    const int64_t k_stride1,    // proj_k.stride(1)
    const int64_t v_stride0,    // proj_v.stride(0)
    const int64_t v_stride1) {  // proj_v.stride(1)
  const int64_t token_idx = static_cast<int64_t>(blockIdx.y);
  if (token_idx >= shared_len) {
    return;
  }

  const int64_t total_elements = kv_heads * head_dim;
  // Threads parallelize along kv_heads * head_dim
  for (int64_t i = threadIdx.x; i < total_elements; i += blockDim.x) {
    const int64_t head_idx = i / head_dim;
    const int64_t dim_idx = i % head_dim;

    // Source index (considering stride for non-contiguous input)
    const int64_t k_src_idx =
        token_idx * k_stride0 + head_idx * k_stride1 + dim_idx;
    const int64_t v_src_idx =
        token_idx * v_stride0 + head_idx * v_stride1 + dim_idx;

    // Destination index (contiguous output)
    const int64_t dst_idx = token_idx * kv_heads * head_dim + i;

    shared_k_cache[dst_idx] = proj_k[k_src_idx];
    shared_v_cache[dst_idx] = proj_v[v_src_idx];
  }
}

}  // namespace

namespace xllm::kernel::cuda {

void prefill_reshape_and_cache(
    torch::Tensor proj_k,  // [shared_len, kv_heads, head_dim]
    torch::Tensor proj_v,  // [shared_len, kv_heads, head_dim]
    torch::Tensor
        shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
    torch::Tensor shared_v_cache) {
  int64_t shared_len = proj_k.size(0);
  shared_k_cache = shared_k_cache.slice(0, 0, shared_len);
  shared_v_cache = shared_v_cache.slice(0, 0, shared_len);

  TORCH_CHECK(proj_k.dim() == 3, "proj_k must be 3-dimensional");
  TORCH_CHECK(proj_v.dim() == 3, "proj_v must be 3-dimensional");
  TORCH_CHECK(shared_k_cache.dim() == 3,
              "shared_k_cache must be 3-dimensional");
  TORCH_CHECK(shared_v_cache.dim() == 3,
              "shared_v_cache must be 3-dimensional");

  const int64_t kv_heads = proj_k.size(1);
  const int64_t head_dim = proj_k.size(2);

  TORCH_CHECK(proj_v.sizes() == proj_k.sizes(),
              "proj_v and proj_k must have same shape");
  TORCH_CHECK(shared_k_cache.size(0) == shared_len &&
                  shared_k_cache.size(1) == kv_heads &&
                  shared_k_cache.size(2) == head_dim,
              "shared_k_cache shape mismatch");
  TORCH_CHECK(shared_v_cache.sizes() == shared_k_cache.sizes(),
              "shared_v_cache and shared_k_cache must have same shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Get stride information for non-contiguous tensor support
  const int64_t k_stride0 = proj_k.stride(0);
  const int64_t k_stride1 = proj_k.stride(1);
  const int64_t v_stride0 = proj_v.stride(0);
  const int64_t v_stride1 = proj_v.stride(1);

  // Launch kernel: one block per token, threads along kv_heads * head_dim
  const int threads_per_block = 128;
  dim3 block_dim(threads_per_block, 1, 1);
  dim3 grid_dim(1, static_cast<unsigned int>(shared_len), 1);

  // Check alignment for vectorization
  DISPATCH_FLOATING_TYPES(
      proj_k.scalar_type(), "prefill_reshape_and_cache_kernel", [&] {
        constexpr int VEC_WIDTH = (std::is_same_v<scalar_t, c10::Half> ||
                                   std::is_same_v<scalar_t, c10::BFloat16>)
                                      ? 8
                                      : 4;  // FP16/BF16: 8, Float: 4

        // Check alignment and vector width requirements
        const auto k_cache_ptr =
            reinterpret_cast<std::uintptr_t>(shared_k_cache.data_ptr());
        const auto v_cache_ptr =
            reinterpret_cast<std::uintptr_t>(shared_v_cache.data_ptr());
        constexpr int alignment_bytes = 16;  // 128-bit alignment

        bool ptrs_aligned = (k_cache_ptr % alignment_bytes == 0) &&
                            (v_cache_ptr % alignment_bytes == 0);
        bool head_dim_multiple = (head_dim % VEC_WIDTH == 0);
        bool use_vectorized = ptrs_aligned && head_dim_multiple;

        if (use_vectorized) {
          prefill_reshape_and_cache_kernel_vec<scalar_t>
              <<<grid_dim, block_dim, 0, stream>>>(
                  proj_k.data_ptr<scalar_t>(),
                  proj_v.data_ptr<scalar_t>(),
                  shared_k_cache.data_ptr<scalar_t>(),
                  shared_v_cache.data_ptr<scalar_t>(),
                  shared_len,
                  kv_heads,
                  head_dim,
                  k_stride0,
                  k_stride1,
                  v_stride0,
                  v_stride1);
        } else {
          prefill_reshape_and_cache_kernel<scalar_t>
              <<<grid_dim, block_dim, 0, stream>>>(
                  proj_k.data_ptr<scalar_t>(),
                  proj_v.data_ptr<scalar_t>(),
                  shared_k_cache.data_ptr<scalar_t>(),
                  shared_v_cache.data_ptr<scalar_t>(),
                  shared_len,
                  kv_heads,
                  head_dim,
                  k_stride0,
                  k_stride1,
                  v_stride0,
                  v_stride1);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
