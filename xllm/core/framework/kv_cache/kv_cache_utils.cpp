/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_cache_utils.h"

#include <algorithm>
#include <cmath>

#include "common/global_flags.h"
#include "framework/kv_cache/kv_cache_shape.h"

namespace xllm {
namespace {

constexpr int64_t kPaddingLinearStateBlocks = 2;
constexpr double kDefaultLinearStateFullKvMemoryRatio = 0.9;

int64_t max_linear_state_blocks(int64_t cache_size_in_bytes,
                                int64_t num_linear_attention_layers,
                                int64_t linear_slot_size,
                                int64_t num_full_attention_layers,
                                int64_t full_attention_block_size) {
  if (linear_slot_size <= 0 || num_linear_attention_layers <= 0) {
    return kPaddingLinearStateBlocks;
  }

  CHECK_GT(cache_size_in_bytes, 0);
  CHECK_GT(full_attention_block_size, 0);
  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_size;
  const int64_t full_cache_bytes_per_block =
      std::max<int64_t>(num_full_attention_layers, 1) *
      full_attention_block_size;
  CHECK_GT(linear_bytes_per_block, 0);
  CHECK_GT(full_cache_bytes_per_block, 0);

  int64_t max_linear_blocks =
      (cache_size_in_bytes - 1) / linear_bytes_per_block;
  // Require enough full-attention blocks to address linear states without
  // starving prefix/full-KV cache: full_blocks + padding >= linear_blocks.
  const int64_t balanced_max_linear_blocks =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_bytes_per_block) /
      (linear_bytes_per_block + full_cache_bytes_per_block);
  max_linear_blocks = std::min(max_linear_blocks, balanced_max_linear_blocks);

  return std::max<int64_t>(max_linear_blocks, kPaddingLinearStateBlocks);
}

int64_t calculate_auto_linear_state_blocks(int64_t cache_size_in_bytes,
                                           int64_t num_linear_attention_layers,
                                           int64_t linear_slot_size) {
  if (linear_slot_size <= 0 || num_linear_attention_layers <= 0) {
    return kPaddingLinearStateBlocks;
  }
  CHECK_GT(cache_size_in_bytes, 0);

  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_size;
  CHECK_GT(linear_bytes_per_block, 0);

  const double linear_memory_fraction =
      kDefaultLinearStateFullKvMemoryRatio /
      (1.0 + kDefaultLinearStateFullKvMemoryRatio);
  const double linear_memory_bytes =
      static_cast<double>(cache_size_in_bytes) * linear_memory_fraction;
  return std::max<int64_t>(
      static_cast<int64_t>(linear_memory_bytes / linear_bytes_per_block),
      kPaddingLinearStateBlocks);
}

}  // namespace

bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval) {
  if (full_attention_interval <= 1) {
    return false;
  }
  return (layer_idx + 1) % full_attention_interval != 0;
}

int64_t calculate_linear_state_blocks(int64_t cache_size_in_bytes,
                                      int64_t num_linear_attention_layers,
                                      int64_t linear_slot_size,
                                      int64_t num_full_attention_layers,
                                      int64_t full_attention_block_size,
                                      const LinearStateCacheOptions& options) {
  const int64_t max_blocks =
      max_linear_state_blocks(cache_size_in_bytes,
                              num_linear_attention_layers,
                              linear_slot_size,
                              num_full_attention_layers,
                              full_attention_block_size);
  if (options.max_linear_state_cache_slots() > 0) {
    const int64_t requested_blocks =
        options.max_linear_state_cache_slots() + kPaddingLinearStateBlocks;
    CHECK_LE(requested_blocks, max_blocks)
        << "max_linear_state_cache_slots requires " << requested_blocks
        << " linear-state blocks, but only " << max_blocks
        << " fit in the configured KV cache budget.";
    return requested_blocks;
  }

  const int64_t auto_blocks = calculate_auto_linear_state_blocks(
      cache_size_in_bytes, num_linear_attention_layers, linear_slot_size);
  return std::max<int64_t>(std::min<int64_t>(auto_blocks, max_blocks),
                           kPaddingLinearStateBlocks);
}

int64_t calculate_linear_state_live_slots(int64_t num_linear_state_blocks,
                                          int64_t max_running_requests) {
  CHECK_GE(num_linear_state_blocks, 0);
  CHECK_GE(max_running_requests, 0);
  if (num_linear_state_blocks <= kPaddingLinearStateBlocks) {
    return num_linear_state_blocks;
  }

  const int64_t active_slot_limit =
      std::max<int64_t>(max_running_requests, 0) + 1;
  const int64_t cacheable_slot_limit =
      std::max<int64_t>(num_linear_state_blocks - kPaddingLinearStateBlocks, 1);
  return std::min(active_slot_limit, cacheable_slot_limit);
}

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  KVCacheTensors tensors;
#if defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  tensors.key_cache = at_npu::native::npu_format_cast(
      torch::empty(
          kv_cache_shape.key_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device())),
      npu_format_type);
  tensors.value_cache = at_npu::native::npu_format_cast(
      torch::empty(
          kv_cache_shape.value_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device())),
      npu_format_type);
#else
  tensors.key_cache = torch::zeros(
      kv_cache_shape.key_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));

  // deepseek_v3 model has no value cache on mlu device
  if (!kv_cache_shape.value_cache_shape().empty()) {
    tensors.value_cache = torch::zeros(
        kv_cache_shape.value_cache_shape(),
        torch::dtype(create_options.dtype()).device(create_options.device()));
  }
#endif
  return tensors;
}

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_index_cache_shape())
      << "index_cache_shape must be initialized.";
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

#if defined(USE_NPU)
  const aclFormat npu_format_type =
      get_npu_kv_cache_format(create_options.model_type());
  tensors.index_cache = at_npu::native::npu_format_cast(
      torch::empty(
          kv_cache_shape.index_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device())),
      npu_format_type);
#else
  tensors.index_cache = torch::zeros(
      kv_cache_shape.index_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
#endif
  return tensors;
}

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
#if !defined(USE_MLU)
  CHECK(!create_options.enable_kv_cache_quant())
      << "KV cache quantization is only supported on MLU backend.";
#endif

  QuantizedKVCacheTensors tensors;
  tensors.kv_cache_tensors =
      create_kv_cache_tensors(kv_cache_shape, create_options);

  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  std::vector<int64_t> key_scale_shape(key_cache_shape.begin(),
                                       key_cache_shape.end() - 1);

  // float32 scale tensor for quantized KV cache (int8)
  tensors.key_cache_scale = torch::zeros(
      key_scale_shape,
      torch::dtype(torch::kFloat32).device(create_options.device()));
  if (!kv_cache_shape.value_cache_shape().empty()) {
    const std::vector<int64_t>& value_cache_shape =
        kv_cache_shape.value_cache_shape();
    std::vector<int64_t> value_scale_shape(value_cache_shape.begin(),
                                           value_cache_shape.end() - 1);
    tensors.value_cache_scale = torch::zeros(
        value_scale_shape,
        torch::dtype(torch::kFloat32).device(create_options.device()));
  }

  return tensors;
}

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options) {
  CHECK(kv_cache_shape.has_conv_cache_shape())
      << "conv_cache_shape must be initialized.";
  CHECK(kv_cache_shape.has_ssm_cache_shape())
      << "ssm_cache_shape must be initialized.";
  LinearAttentionKVCacheTensors tensors;

#if defined(USE_NPU)
  tensors.conv_cache = at_npu::native::npu_format_cast(
      torch::zeros(
          kv_cache_shape.conv_cache_shape(),
          torch::dtype(create_options.dtype()).device(create_options.device())),
      ACL_FORMAT_ND);
  tensors.ssm_cache = at_npu::native::npu_format_cast(
      torch::zeros(kv_cache_shape.ssm_cache_shape(),
                   torch::dtype(create_options.ssm_dtype())
                       .device(create_options.device())),
      ACL_FORMAT_ND);
#else
  tensors.conv_cache = torch::zeros(
      kv_cache_shape.conv_cache_shape(),
      torch::dtype(create_options.dtype()).device(create_options.device()));
  tensors.ssm_cache = torch::zeros(
      kv_cache_shape.ssm_cache_shape(),
      torch::dtype(create_options.ssm_dtype()).device(create_options.device()));
#endif

  return tensors;
}

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type) {
  return model_type == "deepseek_v3" && FLAGS_enable_prefix_cache
             ? ACL_FORMAT_FRACTAL_NZ
             : ACL_FORMAT_ND;
}
#endif

}  // namespace xllm
