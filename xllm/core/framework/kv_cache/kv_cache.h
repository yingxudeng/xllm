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

#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "framework/model/model_input_params.h"

namespace xllm {
class KVCache final {
 public:
  KVCache() = default;
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);
  KVCache(torch::Tensor key_cache,
          torch::Tensor value_cache,
          torch::Tensor index_cache);
  // Constructor for quantized KV cache with scale tensors
  KVCache(torch::Tensor key_cache,
          torch::Tensor value_cache,
          torch::Tensor index_cache,
          torch::Tensor key_cache_scale,
          torch::Tensor value_cache_scale);
  KVCache(torch::Tensor key_cache,
          torch::Tensor value_cache,
          torch::Tensor conv_cache,
          torch::Tensor ssm_cache);
  // DSV4: full constructor with optional indexer_scale, swa and compress states
  KVCache(torch::Tensor key_cache,
          torch::Tensor index_cache,
          torch::Tensor indexer_cache_scale,
          torch::Tensor swa_cache,
          torch::Tensor compress_kv_state,
          torch::Tensor compress_score_state,
          torch::Tensor compress_index_kv_state,
          torch::Tensor compress_index_score_state);
  ~KVCache() = default;

  // TODO: pass in kv_shape and options instead
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;
  torch::Tensor get_index_cache() const;

  // Get scale tensors for quantized KV cache
  std::optional<torch::Tensor> get_k_cache_scale() const;
  std::optional<torch::Tensor> get_v_cache_scale() const;

  torch::Tensor get_conv_cache() const;
  torch::Tensor get_ssm_cache() const;

  // DSV4 cache getters
  torch::Tensor get_indexer_cache_scale() const;
  torch::Tensor get_swa_cache() const;
  torch::Tensor get_compress_kv_state() const;
  torch::Tensor get_compress_score_state() const;
  torch::Tensor get_compress_index_kv_state() const;
  torch::Tensor get_compress_index_score_state() const;
  std::vector<std::vector<int64_t>> get_shapes();

  bool empty() const {
    return !key_cache_.defined() || !value_cache_.defined();
  }

  void swap_blocks(torch::Tensor& src_tensor, torch::Tensor& dst_tensor);

 private:
  torch::Tensor key_cache_;  // [block_size, num_head, head_dim]
  torch::Tensor value_cache_;
  torch::Tensor index_cache_;          // [block_size, num_head, index_head_dim]
  torch::Tensor indexer_cache_scale_;  // [block_size, 1]

  torch::Tensor swa_cache_;             // [window, num_head, head_dim]
  torch::Tensor compress_kv_state_;     // [4*ratio, num_head, 2*head_dim]
  torch::Tensor compress_score_state_;  // [4*ratio, num_head, 2*head_dim]
  torch::Tensor
      compress_index_kv_state_;  // [2*ratio, num_head, 2*index_head_dim]
  torch::Tensor
      compress_index_score_state_;  // [2*ratio, num_head, 2*index_head_dim]

  // scale tensors for quantized KV cache (int8)
  torch::Tensor key_cache_scale_;
  torch::Tensor value_cache_scale_;
  // Convolutional state cache for linear-attention layers (conv_state).
  torch::Tensor conv_cache_;
  // State space model cache for linear-attention layers (ssm_state).
  torch::Tensor ssm_cache_;

};

}  // namespace xllm
