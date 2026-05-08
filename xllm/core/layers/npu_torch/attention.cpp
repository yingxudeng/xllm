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

#include <glog/logging.h>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

torch::Tensor get_npu_kv_seq_lens(const AttentionMetadata& attn_metadata) {
  if (attn_metadata.kv_seq_lens_host.defined()) {
    return attn_metadata.kv_seq_lens_host;
  }
  return attn_metadata.kv_seq_lens;
}

bool has_chunked_prefill_history(const torch::Tensor& q_seq_lens,
                                 const torch::Tensor& kv_seq_lens) {
  CHECK(q_seq_lens.defined()) << "chunked prefill requires q_seq_lens.";
  CHECK(kv_seq_lens.defined()) << "chunked prefill requires kv_seq_lens.";
  CHECK_EQ(q_seq_lens.size(0), kv_seq_lens.size(0));
  const int64_t batch_size = q_seq_lens.size(0);
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const int64_t query_len = q_seq_lens[batch_idx].item<int64_t>();
    const int64_t total_seq_len = kv_seq_lens[batch_idx].item<int64_t>();
    CHECK_GE(total_seq_len, query_len);
    if (total_seq_len > query_len) {
      return true;
    }
  }
  return false;
}

void run_chunked_prefill_paged_attention(const torch::Tensor& query,
                                         torch::Tensor& output,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& block_table,
                                         const torch::Tensor& q_seq_lens,
                                         const torch::Tensor& kv_seq_lens,
                                         float scale,
                                         int64_t num_heads,
                                         int64_t head_size) {
  const int64_t batch_size = q_seq_lens.size(0);
  int64_t query_offset = 0;
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const int64_t query_len = q_seq_lens[batch_idx].item<int64_t>();
    const int64_t total_seq_len = kv_seq_lens[batch_idx].item<int64_t>();
    const int64_t history_len = total_seq_len - query_len;
    CHECK_GE(query_len, 0) << "query_len must be non-negative.";
    CHECK_GE(history_len, 0) << "history_len must be non-negative.";
    if (query_len == 0) {
      continue;
    }

    torch::Tensor block_table_row = block_table[batch_idx].unsqueeze(0);
    for (int64_t token_idx = 0; token_idx < query_len; ++token_idx) {
      torch::Tensor query_token =
          query.narrow(/*dim=*/0, query_offset + token_idx, /*length=*/1)
              .view({1, 1, num_heads, head_size});
      torch::Tensor output_token =
          output.narrow(/*dim=*/0, query_offset + token_idx, /*length=*/1)
              .view({1, 1, num_heads, head_size});
      torch::Tensor token_kv_seq_len =
          torch::tensor({history_len + token_idx + 1}, kv_seq_lens.options());
      xllm::kernel::npu::batch_decode(query_token,
                                      k_cache,
                                      v_cache,
                                      scale,
                                      block_table_row,
                                      token_kv_seq_len,
                                      output_token);
    }
    query_offset += query_len;
  }
}

}  // namespace

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    xllm::kernel::npu::batch_prefill(query,
                                     key,
                                     value,
                                     attn_metadata.attn_mask,
                                     attn_metadata.kv_seq_lens_host,
                                     scale_,
                                     output);
  } else if (attn_metadata.is_chunked_prefill) {
    torch::Tensor kv_seq_lens = get_npu_kv_seq_lens(attn_metadata);
    if (!has_chunked_prefill_history(attn_metadata.q_seq_lens, kv_seq_lens)) {
      key = key.view({-1, num_kv_heads_, head_size_});
      value = value.view({-1, num_kv_heads_, head_size_});
      xllm::kernel::npu::batch_prefill(query,
                                       key,
                                       value,
                                       attn_metadata.attn_mask,
                                       kv_seq_lens,
                                       scale_,
                                       output);
      return;
    }

    CHECK(attn_metadata.block_table.defined())
        << "chunked prefill requires block_table for paged attention.";
    CHECK(v_cache.has_value() && v_cache.value().defined())
        << "chunked prefill requires v_cache.";
    run_chunked_prefill_paged_attention(query,
                                        output,
                                        k_cache,
                                        v_cache.value(),
                                        attn_metadata.block_table,
                                        attn_metadata.q_seq_lens,
                                        kv_seq_lens,
                                        scale_,
                                        num_heads_,
                                        head_size_);
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens = get_npu_kv_seq_lens(attn_metadata);

  if (attn_metadata.paged_attention_tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(
        query,
        k_cache,
        v_cache.value_or(torch::Tensor()),
        scale_,
        attn_metadata.block_table,
        kv_seq_lens,
        attn_metadata.paged_attention_tiling_data,
        output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    attn_metadata.block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
