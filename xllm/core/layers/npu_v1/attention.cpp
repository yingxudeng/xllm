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

#include "attention.h"

#include "core/kernels/npu_v1/ops_npu/npu_ops_api.h"
DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           bool is_prefill,
                                           const torch::Tensor& attn_mask) {
  return AttentionMetadata::build(params, "float", is_prefill, attn_mask);
}

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           const std::string& compute_dtype,
                                           bool is_prefill,
                                           const torch::Tensor& attn_mask) {
  AttentionMetadata attn_metadata;
  attn_metadata.query_start_loc = params.q_seq_lens;
  attn_metadata.seq_start_loc = params.kv_seq_lens;
  attn_metadata.max_query_len = params.q_max_seq_len;
  attn_metadata.max_seq_len = params.kv_max_seq_len;
  attn_metadata.slot_mapping = params.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;
  attn_metadata.attn_mask = attn_mask;

  bool is_start_loc_match = (params.q_seq_lens_vec == params.kv_seq_lens_vec);
  attn_metadata.is_chunked_prefill = is_prefill && !is_start_loc_match;
  attn_metadata.is_prefill = is_prefill && !attn_metadata.is_chunked_prefill;

  // attn_metadata.seq_lens = torch::diff(params.kv_seq_lens);
  attn_metadata.seq_lens = params.kv_seq_lens.to(torch::kCPU);

  if (!attn_metadata.is_prefill) {
    attn_metadata.block_table = params.block_tables;
  }

  return attn_metadata;
}

AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  atb::_npu_reshape_and_cache(
      key, value, k_cache, v_cache, attn_metadata.slot_mapping);

  if (attn_metadata.is_prefill) {
    atb::_npu_flash_attention(query,
                              key,
                              value,
                              attn_metadata.attn_mask,
                              attn_metadata.seq_lens,
                              scale_,
                              num_heads_,
                              num_kv_heads_,
                              output);
  }
  // else if (attn_metadata.is_chunked_prefill) {
  //   tmo::torch_api::flash_attention(query,
  //                                   k_cache,
  //                                   v_cache,
  //                                   output,
  //                                   output_lse,
  //                                   attn_metadata.query_start_loc,
  //                                   attn_metadata.seq_start_loc,
  //                                   std::nullopt /* alibi_slope */,
  //                                   std::nullopt /* attn_bias */,
  //                                   std::nullopt /* q_quant_scale */,
  //                                   std::nullopt /* k_quant_scale */,
  //                                   std::nullopt /* v_quant_scale */,
  //                                   std::nullopt /* out_quant_scale */,
  //                                   attn_metadata.block_table,
  //                                   attn_metadata.max_query_len,
  //                                   attn_metadata.max_seq_len,
  //                                   scale_,
  //                                   true /* is_causal */,
  //                                   sliding_window_,
  //                                   -1,
  //                                   attn_metadata.compute_dtype,
  //                                   false /* return_lse */);
  // }
  else {
    query = query.view({-1, num_heads_, head_size_});
    output = output.view({-1, num_heads_, head_size_});
    atb::_npu_paged_attention(query,
                              k_cache,
                              v_cache,
                              num_kv_heads_,
                              num_heads_,
                              scale_,
                              attn_metadata.block_table,
                              attn_metadata.seq_lens,
                              output);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
