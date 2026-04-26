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

#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

torch::Tensor gather_kv_from_paged_cache(const torch::Tensor& cache,
                                         const torch::Tensor& block_table_row,
                                         int64_t total_seq_len) {
  int64_t block_size = cache.size(1);
  auto positions = torch::arange(total_seq_len,
                                 torch::TensorOptions()
                                     .dtype(torch::kLong)
                                     .device(block_table_row.device()));
  auto logical_block_ids =
      torch::div(positions, block_size, /*rounding_mode=*/"floor");
  auto physical_block_ids =
      block_table_row.to(torch::kLong).index_select(0, logical_block_ids);
  auto token_offsets = torch::remainder(positions, block_size);
  return cache.index({physical_block_ids, token_offsets}).contiguous();
}

torch::Tensor run_chunked_prefill_attention(const torch::Tensor& query,
                                            const torch::Tensor& key,
                                            const torch::Tensor& value,
                                            const torch::Tensor& k_cache,
                                            const torch::Tensor& v_cache,
                                            const torch::Tensor& block_table,
                                            const torch::Tensor& q_seq_lens,
                                            const torch::Tensor& kv_seq_lens,
                                            float scale,
                                            int64_t num_heads,
                                            int64_t num_kv_heads) {
  int64_t head_size = query.size(-1);
  int64_t block_size = k_cache.size(1);
  int64_t batch_size = q_seq_lens.size(0);
  int64_t kv_repeat = num_heads / num_kv_heads;
  int64_t query_offset = 0;

  std::vector<torch::Tensor> outputs;
  outputs.reserve(static_cast<size_t>(batch_size));
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int64_t query_len = q_seq_lens[batch_idx].item<int64_t>();
    int64_t total_seq_len = kv_seq_lens[batch_idx].item<int64_t>();
    int64_t history_len = total_seq_len - query_len;
    CHECK_GE(query_len, 0) << "query_len must be non-negative.";
    CHECK_GE(history_len, 0) << "history_len must be non-negative.";
    if (query_len == 0) {
      continue;
    }

    auto current_k = key.narrow(0, query_offset, query_len);
    auto current_v = value.narrow(0, query_offset, query_len);
    torch::Tensor gathered_k;
    torch::Tensor gathered_v;
    if (history_len > 0) {
      int64_t num_blocks = (history_len + block_size - 1) / block_size;
      auto block_row = block_table[batch_idx].slice(/*dim=*/0, 0, num_blocks);
      gathered_k = torch::cat(
          {gather_kv_from_paged_cache(k_cache, block_row, history_len),
           current_k},
          /*dim=*/0);
      gathered_v = torch::cat(
          {gather_kv_from_paged_cache(v_cache, block_row, history_len),
           current_v},
          /*dim=*/0);
    } else {
      gathered_k = current_k;
      gathered_v = current_v;
    }

    if (kv_repeat > 1) {
      gathered_k = gathered_k.repeat_interleave(kv_repeat, /*dim=*/1);
      gathered_v = gathered_v.repeat_interleave(kv_repeat, /*dim=*/1);
    }

    auto query_chunk = query.narrow(0, query_offset, query_len);
    auto q = query_chunk.transpose(0, 1).to(torch::kFloat32);
    auto k = gathered_k.transpose(0, 1).to(torch::kFloat32);
    auto v = gathered_v.transpose(0, 1).to(torch::kFloat32);
    auto scores = torch::matmul(q, k.transpose(-1, -2)) * scale;

    auto query_positions =
        torch::arange(
            query_len,
            torch::TensorOptions().dtype(torch::kLong).device(query.device()))
            .unsqueeze(1);
    auto key_positions =
        torch::arange(
            total_seq_len,
            torch::TensorOptions().dtype(torch::kLong).device(query.device()))
            .unsqueeze(0);
    auto causal_mask = key_positions <= (query_positions + history_len);
    scores = scores.masked_fill(~causal_mask.unsqueeze(0), -1e30f);

    auto probs = torch::softmax(scores, /*dim=*/-1);
    auto out = torch::matmul(probs, v)
                   .transpose(0, 1)
                   .contiguous()
                   .to(query.dtype())
                   .view({query_len, num_heads, head_size});
    outputs.emplace_back(out);
    query_offset += query_len;
  }

  return torch::cat(outputs, 0);
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
    CHECK(attn_metadata.block_table.defined())
        << "chunked prefill requires block_table for paged KV gather.";
    CHECK(v_cache.has_value() && v_cache.value().defined())
        << "chunked prefill requires v_cache.";
    CHECK(attn_metadata.q_seq_lens.defined())
        << "chunked prefill requires q_seq_lens.";
    CHECK(attn_metadata.kv_seq_lens_host.defined())
        << "chunked prefill requires kv_seq_lens_host.";
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});
    output.copy_(run_chunked_prefill_attention(query,
                                               key,
                                               value,
                                               k_cache,
                                               v_cache.value(),
                                               attn_metadata.block_table,
                                               attn_metadata.q_seq_lens,
                                               attn_metadata.kv_seq_lens_host,
                                               scale_,
                                               num_heads_,
                                               num_kv_heads_));
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

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
