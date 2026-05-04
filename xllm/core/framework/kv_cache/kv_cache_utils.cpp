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

#include "common/global_flags.h"
#include "framework/kv_cache/kv_cache_shape.h"

namespace xllm {
namespace {

constexpr int64_t kPaddingLinearStateBlocks = 2;

int64_t clamp_linear_state_blocks(int64_t requested_blocks,
                                  int64_t cache_size_in_bytes,
                                  int64_t num_linear_attention_layers,
                                  int64_t linear_slot_size,
                                  int64_t num_full_attention_layers,
                                  int64_t full_attention_block_size,
                                  int64_t min_full_kv_cache_blocks) {
  if (linear_slot_size <= 0 || num_linear_attention_layers <= 0) {
    return requested_blocks;
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

  const int64_t required_full_kv_blocks =
      std::max<int64_t>(min_full_kv_cache_blocks, 0);

  int64_t max_linear_blocks =
      (cache_size_in_bytes - 1) / linear_bytes_per_block;
  if (required_full_kv_blocks > 0) {
    CHECK_LE(linear_bytes_per_block,
             cache_size_in_bytes / kPaddingLinearStateBlocks)
        << "linear-state padding cannot fit in the cache budget.";
    const int64_t bytes_after_padding_linear =
        cache_size_in_bytes -
        kPaddingLinearStateBlocks * linear_bytes_per_block;
    const int64_t max_full_blocks_with_padding =
        bytes_after_padding_linear / full_cache_bytes_per_block;
    CHECK_LE(required_full_kv_blocks, max_full_blocks_with_padding)
        << "min_full_kv_cache_blocks cannot be preserved with mandatory "
           "linear-state padding.";
    const int64_t bytes_after_min_full_kv =
        cache_size_in_bytes -
        required_full_kv_blocks * full_cache_bytes_per_block;
    max_linear_blocks = std::min(
        max_linear_blocks, bytes_after_min_full_kv / linear_bytes_per_block);
  }

  // Require enough full-attention blocks to address linear states without
  // starving prefix/full-KV cache: full_blocks + padding >= linear_blocks.
  const int64_t balanced_max_linear_blocks =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_bytes_per_block) /
      (linear_bytes_per_block + full_cache_bytes_per_block);
  max_linear_blocks = std::min(max_linear_blocks, balanced_max_linear_blocks);

  return std::max<int64_t>(
      std::min<int64_t>(requested_blocks, max_linear_blocks),
      kPaddingLinearStateBlocks);
}

int64_t calculate_auto_linear_state_blocks(
    int64_t cache_size_in_bytes,
    int64_t num_linear_attention_layers,
    int64_t linear_slot_size,
    int64_t num_full_attention_layers,
    int64_t full_attention_block_size,
    const LinearStateCacheOptions& options) {
  if (linear_slot_size <= 0 || num_linear_attention_layers <= 0) {
    return kPaddingLinearStateBlocks;
  }
  CHECK_GT(cache_size_in_bytes, 0);
  CHECK_GT(full_attention_block_size, 0);
  CHECK_GT(options.linear_state_full_kv_memory_ratio(), 0.0);

  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_size;
  const int64_t full_cache_bytes_per_block =
      std::max<int64_t>(num_full_attention_layers, 1) *
      full_attention_block_size;
  CHECK_GT(linear_bytes_per_block, 0);
  CHECK_GT(full_cache_bytes_per_block, 0);

  const double linear_memory_fraction =
      options.linear_state_full_kv_memory_ratio() /
      (1.0 + options.linear_state_full_kv_memory_ratio());
  CHECK(std::isfinite(linear_memory_fraction))
      << "linear_state_full_kv_memory_ratio is too large.";
  const double linear_memory_bytes =
      static_cast<double>(cache_size_in_bytes) * linear_memory_fraction;
  const int64_t requested_blocks = std::max<int64_t>(
      static_cast<int64_t>(linear_memory_bytes / linear_bytes_per_block),
      kPaddingLinearStateBlocks);
  return clamp_linear_state_blocks(requested_blocks,
                                   cache_size_in_bytes,
                                   num_linear_attention_layers,
                                   linear_slot_size,
                                   num_full_attention_layers,
                                   full_attention_block_size,
                                   options.min_full_kv_cache_blocks());
}

}  // namespace

bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval) {
  if (full_attention_interval <= 1) {
    return false;
  }
  return (layer_idx + 1) % full_attention_interval != 0;
}

int64_t calculate_linear_state_blocks(int64_t max_seqs_per_batch,
                                      int64_t cache_size_in_bytes,
                                      int64_t num_linear_attention_layers,
                                      int64_t linear_slot_size,
                                      int64_t num_full_attention_layers,
                                      int64_t full_attention_block_size) {
  const int64_t requested_blocks =
      std::max<int64_t>(max_seqs_per_batch, 0) + kPaddingLinearStateBlocks;
  return clamp_linear_state_blocks(requested_blocks,
                                   cache_size_in_bytes,
                                   num_linear_attention_layers,
                                   linear_slot_size,
                                   num_full_attention_layers,
                                   full_attention_block_size,
                                   /*min_full_kv_cache_blocks=*/0);
}

int64_t calculate_linear_state_blocks(int64_t max_seqs_per_batch,
                                      int64_t cache_size_in_bytes,
                                      int64_t num_linear_attention_layers,
                                      int64_t linear_slot_size,
                                      int64_t num_full_attention_layers,
                                      int64_t full_attention_block_size,
                                      const LinearStateCacheOptions& options) {
  if (options.policy() == LinearStateCachePolicy::AUTO) {
    return calculate_auto_linear_state_blocks(cache_size_in_bytes,
                                              num_linear_attention_layers,
                                              linear_slot_size,
                                              num_full_attention_layers,
                                              full_attention_block_size,
                                              options);
  }

  if (options.policy() == LinearStateCachePolicy::FIXED) {
    const int64_t requested_blocks =
        options.max_linear_state_cache_slots() + kPaddingLinearStateBlocks;
    return clamp_linear_state_blocks(requested_blocks,
                                     cache_size_in_bytes,
                                     num_linear_attention_layers,
                                     linear_slot_size,
                                     num_full_attention_layers,
                                     full_attention_block_size,
                                     options.min_full_kv_cache_blocks());
  }

  const int64_t requested_blocks =
      std::max<int64_t>(max_seqs_per_batch, 0) + kPaddingLinearStateBlocks;
  return clamp_linear_state_blocks(requested_blocks,
                                   cache_size_in_bytes,
                                   num_linear_attention_layers,
                                   linear_slot_size,
                                   num_full_attention_layers,
                                   full_attention_block_size,
                                   options.min_full_kv_cache_blocks());
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
