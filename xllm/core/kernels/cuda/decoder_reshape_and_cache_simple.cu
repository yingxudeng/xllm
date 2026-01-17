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

#include "cuda_ops_api.h"
#include "utils.h"

namespace {

// Simple decoder reshape and cache kernel.
// Copies proj_k and proj_v into unshared_k_cache / unshared_v_cache.
// Inputs:
//   proj_k           : [T, kv_heads, head_dim]
//   proj_v           : [T, kv_heads, head_dim]
// Outputs:
//   unshared_k_cache : [T, kv_heads, head_dim]
//   unshared_v_cache : [T, kv_heads, head_dim]
// where T = batch_size * beam_size
template <typename scalar_t>
__global__ void decoder_reshape_and_cache_simple_kernel(
    const scalar_t* __restrict__ proj_k,      // [T, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,      // [T, kv_heads, head_dim]
    scalar_t* __restrict__ unshared_k_cache,  // [T, kv_heads, head_dim]
    scalar_t* __restrict__ unshared_v_cache,  // [T, kv_heads, head_dim]
    const int64_t T,                // Total tokens (batch_size * beam_size)
    const int64_t kv_heads,         // Number of kv heads
    const int64_t head_dim,         // Head dimension
    const int64_t k_stride0,        // proj_k.stride(0)
    const int64_t k_stride1,        // proj_k.stride(1)
    const int64_t k_stride2,        // proj_k.stride(2)
    const int64_t v_stride0,        // proj_v.stride(0)
    const int64_t v_stride1,        // proj_v.stride(1)
    const int64_t v_stride2,        // proj_v.stride(2)
    const int64_t cache_stride0,    // unshared_k_cache.stride(0)
    const int64_t cache_stride1,    // unshared_k_cache.stride(1)
    const int64_t cache_stride2) {  // unshared_k_cache.stride(2)
  const int64_t total_elements = T * kv_heads;
  const int64_t idx = static_cast<int64_t>(blockIdx.y);

  if (idx >= total_elements) {
    return;
  }

  // Decode flattened index -> (token_idx, kv_head_idx)
  const int64_t token_idx = idx / kv_heads;
  const int64_t kv_head_idx = idx % kv_heads;

  // Compute base indices using stride
  const int64_t k_src_base = token_idx * k_stride0 + kv_head_idx * k_stride1;
  const int64_t v_src_base = token_idx * v_stride0 + kv_head_idx * v_stride1;
  const int64_t cache_dst_base =
      token_idx * cache_stride0 + kv_head_idx * cache_stride1;

  // Copy the full head_dim with threads parallelizing along D.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    unshared_k_cache[cache_dst_base + d * cache_stride2] =
        proj_k[k_src_base + d * k_stride2];
    unshared_v_cache[cache_dst_base + d * cache_stride2] =
        proj_v[v_src_base + d * v_stride2];
  }
}

}  // namespace

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache_simple(torch::Tensor proj_k,
                                      torch::Tensor proj_v,
                                      torch::Tensor unshared_k_cache,
                                      torch::Tensor unshared_v_cache) {
  TORCH_CHECK(proj_k.dim() == 3, "proj_k must be 3-dimensional");
  TORCH_CHECK(proj_v.dim() == 3, "proj_v must be 3-dimensional");
  TORCH_CHECK(unshared_k_cache.dim() == 3,
              "unshared_k_cache must be 3-dimensional");
  TORCH_CHECK(unshared_v_cache.dim() == 3,
              "unshared_v_cache must be 3-dimensional");

  const int64_t T = proj_k.size(0);
  const int64_t kv_heads = proj_k.size(1);
  const int64_t head_dim = proj_k.size(2);

  TORCH_CHECK(proj_v.sizes() == proj_k.sizes(),
              "proj_v and proj_k must have same shape");
  TORCH_CHECK(unshared_k_cache.sizes() == proj_k.sizes(),
              "unshared_k_cache must have same shape as proj_k");
  TORCH_CHECK(unshared_v_cache.sizes() == proj_k.sizes(),
              "unshared_v_cache must have same shape as proj_k");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Get stride information for non-contiguous tensor support
  const int64_t k_stride0 = proj_k.stride(0);
  const int64_t k_stride1 = proj_k.stride(1);
  const int64_t k_stride2 = proj_k.stride(2);
  const int64_t v_stride0 = proj_v.stride(0);
  const int64_t v_stride1 = proj_v.stride(1);
  const int64_t v_stride2 = proj_v.stride(2);
  const int64_t cache_stride0 = unshared_k_cache.stride(0);
  const int64_t cache_stride1 = unshared_k_cache.stride(1);
  const int64_t cache_stride2 = unshared_k_cache.stride(2);

  // Launch kernel: one block per (token, kv_head), threads along head_dim.
  const int64_t total_elements = T * kv_heads;
  const int threads_per_block = 128;
  dim3 block_dim(threads_per_block, 1, 1);
  dim3 grid_dim(1, static_cast<unsigned int>(total_elements), 1);

  DISPATCH_FLOATING_TYPES(
      proj_k.scalar_type(), "decoder_reshape_and_cache_simple_kernel", [&] {
        decoder_reshape_and_cache_simple_kernel<scalar_t>
            <<<grid_dim, block_dim, 0, stream>>>(
                proj_k.data_ptr<scalar_t>(),
                proj_v.data_ptr<scalar_t>(),
                unshared_k_cache.data_ptr<scalar_t>(),
                unshared_v_cache.data_ptr<scalar_t>(),
                T,
                kv_heads,
                head_dim,
                k_stride0,
                k_stride1,
                k_stride2,
                v_stride0,
                v_stride1,
                v_stride2,
                cache_stride0,
                cache_stride1,
                cache_stride2);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
