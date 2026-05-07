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

#include <algorithm>
#include <limits>
#include <vector>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

namespace {

bool has_chunked_prefill_history(const AttentionMetadata& attn_metadata) {
  CHECK(attn_metadata.q_seq_lens.defined());
  CHECK(attn_metadata.kv_seq_lens_host.defined());
  const torch::Tensor q_lens_host =
      attn_metadata.q_seq_lens.to(torch::kCPU).to(torch::kInt32).contiguous();
  const torch::Tensor kv_lens_host =
      attn_metadata.kv_seq_lens_host.to(torch::kCPU)
          .to(torch::kInt32)
          .contiguous();
  CHECK_EQ(q_lens_host.numel(), kv_lens_host.numel());
  for (int64_t i = 0; i < q_lens_host.numel(); ++i) {
    if (kv_lens_host[i].item<int32_t>() > q_lens_host[i].item<int32_t>()) {
      return true;
    }
  }
  return false;
}

void copy_cache_tokens_to_dense(const torch::Tensor& cache,
                                const torch::Tensor& block_table,
                                int64_t seq_idx,
                                int64_t num_tokens,
                                torch::Tensor& dense) {
  const int64_t block_size = cache.size(1);
  int64_t tokens_copied = 0;
  while (tokens_copied < num_tokens) {
    const int64_t block_idx = tokens_copied / block_size;
    const int64_t block_offset = tokens_copied % block_size;
    const int64_t copy_len = std::min<int64_t>(block_size - block_offset,
                                               num_tokens - tokens_copied);
    const int64_t physical_block =
        block_table[seq_idx][block_idx].item<int64_t>();
    CHECK_GE(physical_block, 0);
    dense.slice(/*dim=*/0, tokens_copied, tokens_copied + copy_len)
        .copy_(cache.select(/*dim=*/0, physical_block)
                   .slice(/*dim=*/0, block_offset, block_offset + copy_len));
    tokens_copied += copy_len;
  }
}

torch::Tensor build_chunked_prefill_causal_mask(int64_t q_len,
                                                int64_t kv_len,
                                                const torch::Device& device) {
  const int64_t history_len = kv_len - q_len;
  CHECK_GE(history_len, 0);
  torch::Tensor q_positions =
      torch::arange(q_len,
                    torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1) +
      history_len;
  torch::Tensor kv_positions = torch::arange(
      kv_len, torch::TensorOptions().dtype(torch::kLong).device(device));
  return kv_positions.unsqueeze(0) <= q_positions;
}

void chunked_prefill_dense_attention(const torch::Tensor& query,
                                     const torch::Tensor& current_key,
                                     const torch::Tensor& current_value,
                                     torch::Tensor& output,
                                     const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache,
                                     float scale,
                                     const AttentionMetadata& attn_metadata) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(current_key.dim(), 3);
  CHECK_EQ(current_value.dim(), 3);
  CHECK_EQ(output.dim(), 3);
  CHECK(attn_metadata.block_table.defined());
  CHECK(attn_metadata.q_seq_lens.defined());
  CHECK(attn_metadata.kv_seq_lens_host.defined());

  const int64_t num_sequences = attn_metadata.q_seq_lens.size(0);
  CHECK_EQ(attn_metadata.kv_seq_lens_host.size(0), num_sequences);
  CHECK_EQ(attn_metadata.block_table.size(0), num_sequences);

  const torch::Tensor q_lens_host =
      attn_metadata.q_seq_lens.to(torch::kCPU).to(torch::kInt32).contiguous();
  const torch::Tensor kv_lens_host =
      attn_metadata.kv_seq_lens_host.to(torch::kCPU)
          .to(torch::kInt32)
          .contiguous();

  int64_t token_offset = 0;
  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const int64_t q_len = q_lens_host[seq_idx].item<int32_t>();
    const int64_t kv_len = kv_lens_host[seq_idx].item<int32_t>();
    CHECK_GE(q_len, 0);
    CHECK_GE(kv_len, q_len);
    if (q_len == 0) {
      continue;
    }

    const int64_t history_len = kv_len - q_len;
    torch::Tensor key = torch::empty({kv_len, k_cache.size(2), k_cache.size(3)},
                                     query.options());
    torch::Tensor value = torch::empty(
        {kv_len, v_cache.size(2), v_cache.size(3)}, query.options());
    if (history_len > 0) {
      torch::Tensor history_key = key.slice(/*dim=*/0, 0, history_len);
      torch::Tensor history_value = value.slice(/*dim=*/0, 0, history_len);
      copy_cache_tokens_to_dense(k_cache,
                                 attn_metadata.block_table,
                                 seq_idx,
                                 history_len,
                                 history_key);
      copy_cache_tokens_to_dense(v_cache,
                                 attn_metadata.block_table,
                                 seq_idx,
                                 history_len,
                                 history_value);
    }
    key.slice(/*dim=*/0, history_len, kv_len)
        .copy_(
            current_key.slice(/*dim=*/0, token_offset, token_offset + q_len));
    value.slice(/*dim=*/0, history_len, kv_len)
        .copy_(
            current_value.slice(/*dim=*/0, token_offset, token_offset + q_len));

    torch::Tensor seq_query =
        query.slice(/*dim=*/0, token_offset, token_offset + q_len)
            .to(torch::kFloat32);
    torch::Tensor seq_key = key.to(torch::kFloat32);
    torch::Tensor seq_value = value.to(torch::kFloat32);
    if (seq_key.size(1) != seq_query.size(1)) {
      const int64_t repeat_factor = seq_query.size(1) / seq_key.size(1);
      CHECK_EQ(seq_key.size(1) * repeat_factor, seq_query.size(1));
      seq_key = seq_key.repeat_interleave(repeat_factor, /*dim=*/1);
      seq_value = seq_value.repeat_interleave(repeat_factor, /*dim=*/1);
    }

    torch::Tensor scores = torch::einsum("qhd,khd->hqk", {seq_query, seq_key});
    scores = scores * static_cast<double>(scale);
    torch::Tensor causal_mask =
        build_chunked_prefill_causal_mask(q_len, kv_len, query.device());
    scores = scores.masked_fill(causal_mask.logical_not().unsqueeze(0),
                                -std::numeric_limits<float>::infinity());
    torch::Tensor probs = torch::softmax(scores, /*dim=*/-1);
    torch::Tensor seq_output =
        torch::einsum("hqk,khd->qhd", {probs, seq_value}).to(output.dtype());
    output.slice(/*dim=*/0, token_offset, token_offset + q_len)
        .copy_(seq_output);
    token_offset += q_len;
  }
  CHECK_EQ(token_offset, query.size(0));
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
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});
    if (!has_chunked_prefill_history(attn_metadata)) {
      xllm::kernel::npu::batch_prefill(query,
                                       key,
                                       value,
                                       attn_metadata.attn_mask,
                                       attn_metadata.kv_seq_lens_host,
                                       scale_,
                                       output);
      return;
    }
    chunked_prefill_dense_attention(query,
                                    key,
                                    value,
                                    output,
                                    k_cache,
                                    v_cache.value(),
                                    scale_,
                                    attn_metadata);
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
