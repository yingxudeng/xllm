/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "air_topk_last_dim.h"
#include "topk_last_dim.cuh"

namespace xllm::kernel::cuda {

namespace {

struct WorkspaceCache {
  torch::Tensor buffer;
  size_t capacity = 0;
  int device_index = -1;
};

torch::Tensor get_workspace(size_t required_bytes,
                            const torch::Device& device) {
  thread_local std::unordered_map<int, WorkspaceCache> caches;
  int dev_idx = device.index();
  auto& cache = caches[dev_idx];

  if (cache.capacity < required_bytes || cache.device_index != dev_idx) {
    cache.buffer = torch::empty(
        {static_cast<int64_t>(required_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    cache.capacity = required_bytes;
    cache.device_index = dev_idx;
  }
  return cache.buffer;
}

struct WorkspaceSizeKey {
  int32_t batch;
  int32_t len;
  int32_t k;
  int dtype_code;  // 0=float, 1=bf16, 2=fp16
  bool sorted;

  bool operator==(const WorkspaceSizeKey& other) const {
    return batch == other.batch && len == other.len && k == other.k &&
           dtype_code == other.dtype_code && sorted == other.sorted;
  }
};

struct WorkspaceSizeKeyHash {
  size_t operator()(const WorkspaceSizeKey& key) const {
    size_t h = std::hash<int32_t>{}(key.batch);
    h ^= std::hash<int32_t>{}(key.len) << 1;
    h ^= std::hash<int32_t>{}(key.k) << 2;
    h ^= std::hash<int>{}(key.dtype_code) << 3;
    h ^= std::hash<bool>{}(key.sorted) << 4;
    return h;
  }
};

static inline std::
    unordered_map<WorkspaceSizeKey, size_t, WorkspaceSizeKeyHash>&
    get_workspace_size_cache() {
  thread_local std::
      unordered_map<WorkspaceSizeKey, size_t, WorkspaceSizeKeyHash>
          size_cache;
  return size_cache;
}

size_t get_cached_workspace_size(int32_t batch,
                                 int32_t len,
                                 int32_t k,
                                 int dtype_code,
                                 bool sorted) {
  WorkspaceSizeKey key{batch, len, k, dtype_code, sorted};
  auto& size_cache = get_workspace_size_cache();
  auto it = size_cache.find(key);
  if (it != size_cache.end()) {
    return it->second;
  }
  return 0;
}

void cache_workspace_size(int32_t batch,
                          int32_t len,
                          int32_t k,
                          int dtype_code,
                          bool sorted,
                          size_t size) {
  WorkspaceSizeKey key{batch, len, k, dtype_code, sorted};
  auto& size_cache = get_workspace_size_cache();
  size_cache[key] = size;
}

template <typename T>
void run_topk_with_cache(const T* in_ptr,
                         T* out_val,
                         int32_t* out_idx,
                         int32_t batch,
                         int32_t len,
                         int32_t k,
                         int dtype_code,
                         bool largest,
                         bool sorted_by_value,
                         const torch::Device& device,
                         cudaStream_t stream) {
  size_t workspace_bytes =
      get_cached_workspace_size(batch, len, k, dtype_code, sorted_by_value);
  if (workspace_bytes == 0) {
    workspace_bytes = reduce_topk::invokeComputeTopkLastDimWorkspaceSize<T>(
        static_cast<SizeType32>(batch),
        static_cast<SizeType32>(len),
        static_cast<SizeType32>(k),
        largest,
        sorted_by_value);
    cache_workspace_size(
        batch, len, k, dtype_code, sorted_by_value, workspace_bytes);
  }

  auto workspace = get_workspace(workspace_bytes, device);
  reduce_topk::invokeTopkLastDim<T>(static_cast<SizeType32>(batch),
                                    static_cast<SizeType32>(len),
                                    static_cast<SizeType32>(k),
                                    largest,
                                    in_ptr,
                                    out_val,
                                    out_idx,
                                    workspace.data_ptr<uint8_t>(),
                                    stream,
                                    sorted_by_value);
}

struct OutputCache {
  torch::Tensor values;
  torch::Tensor indices;
  int64_t batch = 0;
  int64_t k = 0;
  int device_index = -1;
  torch::ScalarType dtype = torch::kFloat32;
};

std::pair<torch::Tensor, torch::Tensor> get_cached_output(
    int64_t batch,
    int64_t k,
    const torch::Device& device,
    torch::ScalarType dtype) {
  thread_local std::unordered_map<int, OutputCache> caches;
  int dev_idx = device.index();
  auto& cache = caches[dev_idx];

  bool need_realloc = (cache.batch < batch || cache.k < k ||
                       cache.device_index != dev_idx || cache.dtype != dtype);
  if (need_realloc) {
    cache.values = torch::empty(
        {batch, k}, torch::TensorOptions().dtype(dtype).device(device));
    cache.indices = torch::empty(
        {batch, k}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    cache.batch = batch;
    cache.k = k;
    cache.device_index = dev_idx;
    cache.dtype = dtype;
  }

  if (cache.batch == batch && cache.k == k) {
    return {cache.values, cache.indices};
  }
  return {cache.values.slice(0, 0, batch).slice(1, 0, k),
          cache.indices.slice(0, 0, batch).slice(1, 0, k)};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> air_topk_last_dim(
    const torch::Tensor& input,
    int32_t k,
    bool largest,
    bool sorted_by_value) {
  TORCH_CHECK(input.is_cuda(), "air_topk_last_dim: input must be CUDA");
  TORCH_CHECK(input.dim() == 2, "air_topk_last_dim: input must be 2D [B, L]");
  TORCH_CHECK(k > 0, "air_topk_last_dim: k must be > 0");

  const int64_t batch64 = input.size(0);
  const int64_t len64 = input.size(1);
  TORCH_CHECK(batch64 >= 0 && batch64 <= INT32_MAX,
              "air_topk_last_dim: batch too large");
  TORCH_CHECK(len64 > 0 && len64 <= INT32_MAX,
              "air_topk_last_dim: len too large");
  TORCH_CHECK(k <= len64, "air_topk_last_dim: k must be <= len");

  if (batch64 == 0) {
    auto empty_vals = torch::empty({batch64, k},
                                   torch::TensorOptions()
                                       .dtype(input.scalar_type())
                                       .device(input.device()));
    auto empty_idx = torch::empty(
        {batch64, k},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    return {empty_vals, empty_idx};
  }

  const int32_t batch = static_cast<int32_t>(batch64);
  const int32_t len = static_cast<int32_t>(len64);

  c10::cuda::CUDAGuard device_guard(input.device());
  auto in = input.contiguous();
  auto [values, indices] = get_cached_output(
      batch64, static_cast<int64_t>(k), in.device(), in.scalar_type());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto dtype = in.scalar_type();

  if (dtype == torch::kFloat32) {
    run_topk_with_cache<float>(in.data_ptr<float>(),
                               values.data_ptr<float>(),
                               indices.data_ptr<int32_t>(),
                               batch,
                               len,
                               k,
                               0,
                               largest,
                               sorted_by_value,
                               in.device(),
                               stream);
  } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
    using half = __half;
    run_topk_with_cache<half>(
        reinterpret_cast<const half*>(in.data_ptr<at::Half>()),
        reinterpret_cast<half*>(values.data_ptr<at::Half>()),
        indices.data_ptr<int32_t>(),
        batch,
        len,
        k,
        2,
        largest,
        sorted_by_value,
        in.device(),
        stream);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    run_topk_with_cache<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
        indices.data_ptr<int32_t>(),
        batch,
        len,
        k,
        1,
        largest,
        sorted_by_value,
        in.device(),
        stream);
#else
    auto in_f = in.to(torch::kFloat32);
    auto out_f = values.to(torch::kFloat32);
    run_topk_with_cache<float>(in_f.data_ptr<float>(),
                               out_f.data_ptr<float>(),
                               indices.data_ptr<int32_t>(),
                               batch,
                               len,
                               k,
                               0,
                               largest,
                               sorted_by_value,
                               in.device(),
                               stream);
    values.copy_(out_f);
#endif
  } else {
    TORCH_CHECK(false, "air_topk_last_dim: unsupported dtype");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, indices};
}

}  // namespace xllm::kernel::cuda
