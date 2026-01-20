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

#include "core/common/global_flags.h"
#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/attention_runner.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {
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
    KVCache& kv_cache,
    std::optional<torch::Tensor> output) {
  if (!output.has_value()) {
    output = torch::empty_like(query);
  }
  auto output_tensor = output.value();
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output_tensor = output_tensor.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output_tensor, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output_tensor = output_tensor.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // maybe we need to update shared attn state before execute attention,
  // currently we update flashinfer step_wise_attn_state_ at layer 0.
  bool causal = attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(50) << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_plan_info(
        attn_metadata.plan_info,
        causal ? xllm::kernel::cuda::determine_attention_backend(
                     /*pos_encoding_mode=*/0,
                     /*use_fp16_qk_reduction=*/false,
                     /*use_custom_mask=*/false)
               : "fa2",
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output_tensor.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size=*/k_cache.size(1),
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/attn_metadata.enable_cuda_graph,
        /*causal=*/causal,
        /*use_tensor_core=*/attn_metadata.use_tensor_core);
  }

  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key;
  reshape_paged_cache_params.value = value;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  xllm::kernel::AttentionParams attention_params(attn_metadata);
  attention_params.query = query;
  attention_params.output = output_tensor;
  attention_params.output_lse = output_lse;
  // attention_params.max_seq_len = attn_metadata.max_seq_len;
  attention_params.window_size_left = sliding_window_;
  attention_params.scale = scale_;
  // for flashinfer
  attention_params.float_workspace_buffer =
      attn_metadata.float_workspace_buffer;
  attention_params.int_workspace_buffer = attn_metadata.int_workspace_buffer;
  attention_params.page_locked_int_workspace_buffer =
      attn_metadata.page_locked_int_workspace_buffer;

  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";
  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    attention_params.query = query;
    attention_params.output = output_tensor;
    attention_params.k_cache = k_cache;
    attention_params.v_cache = v_cache;

    // Log detailed information about plan_info and workspace buffers
    if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
      VLOG(kGraphExecutorLogVerboseLevel) << "=== batch_decode parameters ===";
      VLOG(kGraphExecutorLogVerboseLevel) << "query shape: " << query.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "output shape: " << output_tensor.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "k_cache shape: " << k_cache.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "v_cache shape: " << v_cache.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "paged_kv_indptr shape: " << attn_metadata.paged_kv_indptr.sizes();
      if (attn_metadata.paged_kv_indptr.defined()) {
        torch::Tensor paged_kv_indptr_cpu =
            attn_metadata.paged_kv_indptr.to(torch::kCPU);
        VLOG(kGraphExecutorLogVerboseLevel)
            << "paged_kv_indptr values: " << paged_kv_indptr_cpu;
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "paged_kv_indices shape: "
          << attn_metadata.paged_kv_indices.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "paged_kv_last_page_len shape: "
          << attn_metadata.paged_kv_last_page_len.sizes();
      if (attn_metadata.paged_kv_last_page_len.defined()) {
        torch::Tensor paged_kv_last_page_len_cpu =
            attn_metadata.paged_kv_last_page_len.to(torch::kCPU);
        VLOG(kGraphExecutorLogVerboseLevel)
            << "paged_kv_last_page_len values: " << paged_kv_last_page_len_cpu;
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "kv_seq_lens shape: " << attn_metadata.kv_seq_lens.sizes();
      if (attn_metadata.kv_seq_lens.defined()) {
        torch::Tensor kv_seq_lens_cpu =
            attn_metadata.kv_seq_lens.to(torch::kCPU);
        VLOG(kGraphExecutorLogVerboseLevel)
            << "kv_seq_lens values: " << kv_seq_lens_cpu;
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "plan_info shape: " << attn_metadata.plan_info->plan_info.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "plan_info dtype: "
          << attn_metadata.plan_info->plan_info.scalar_type();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "plan_info device: "
          << attn_metadata.plan_info->plan_info.device();
      if (attn_metadata.plan_info->plan_info.defined()) {
        torch::Tensor plan_info_cpu =
            attn_metadata.plan_info->plan_info.to(torch::kCPU);
        VLOG(kGraphExecutorLogVerboseLevel)
            << "plan_info values: " << plan_info_cpu;
        // Decode plan_info structure for decode mode
        // plan_info is a vector of int64_t: [padded_batch_size, v_offset,
        // s_offset, request_indices_offset, kv_tile_indices_offset,
        // o_indptr_offset, block_valid_mask_offset, kv_chunk_size_ptr_offset,
        // enable_cuda_graph, split_kv]
        if (plan_info_cpu.numel() >= 10) {
          VLOG(kGraphExecutorLogVerboseLevel) << "plan_info decoded:";
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  padded_batch_size: " << plan_info_cpu[0].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  v_offset: " << plan_info_cpu[1].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  s_offset: " << plan_info_cpu[2].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  request_indices_offset: "
              << plan_info_cpu[3].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  kv_tile_indices_offset: "
              << plan_info_cpu[4].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  o_indptr_offset: " << plan_info_cpu[5].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  block_valid_mask_offset: "
              << plan_info_cpu[6].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  kv_chunk_size_ptr_offset: "
              << plan_info_cpu[7].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  enable_cuda_graph: " << plan_info_cpu[8].item<int64_t>();
          VLOG(kGraphExecutorLogVerboseLevel)
              << "  split_kv: " << plan_info_cpu[9].item<int64_t>();
        }
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "int_workspace_buffer shape: "
          << attention_params.int_workspace_buffer.sizes();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "int_workspace_buffer numel: "
          << attention_params.int_workspace_buffer.numel();
      VLOG(kGraphExecutorLogVerboseLevel)
          << "int_workspace_buffer dtype: "
          << attention_params.int_workspace_buffer.scalar_type();
      if (attn_metadata.plan_info->plan_info.defined() &&
          attention_params.int_workspace_buffer.defined()) {
        torch::Tensor plan_info_cpu =
            attn_metadata.plan_info->plan_info.to(torch::kCPU);
        if (plan_info_cpu.numel() >= 10) {
          int64_t request_indices_offset = plan_info_cpu[3].item<int64_t>();
          int64_t kv_tile_indices_offset = plan_info_cpu[4].item<int64_t>();
          int64_t o_indptr_offset = plan_info_cpu[5].item<int64_t>();
          int64_t padded_batch_size = plan_info_cpu[0].item<int64_t>();

          // Check int_workspace_buffer content at specified offsets
          torch::Tensor int_workspace_cpu =
              attention_params.int_workspace_buffer.to(torch::kCPU);
          VLOG(kGraphExecutorLogVerboseLevel)
              << "int_workspace_buffer total size: "
              << int_workspace_cpu.numel();

          if (request_indices_offset >= 0 &&
              request_indices_offset < int_workspace_cpu.numel()) {
            int64_t num_request_indices =
                padded_batch_size > 0 ? padded_batch_size : 1;
            if (request_indices_offset + num_request_indices <=
                int_workspace_cpu.numel()) {
              torch::Tensor request_indices = int_workspace_cpu.slice(
                  0,
                  request_indices_offset,
                  request_indices_offset + num_request_indices);
              VLOG(kGraphExecutorLogVerboseLevel)
                  << "request_indices (offset=" << request_indices_offset
                  << ", size=" << num_request_indices
                  << "): " << request_indices;
            }
          }

          if (o_indptr_offset >= 0 &&
              o_indptr_offset < int_workspace_cpu.numel()) {
            int64_t num_o_indptr =
                padded_batch_size > 0 ? padded_batch_size + 1 : 2;
            if (o_indptr_offset + num_o_indptr <= int_workspace_cpu.numel()) {
              torch::Tensor o_indptr = int_workspace_cpu.slice(
                  0, o_indptr_offset, o_indptr_offset + num_o_indptr);
              VLOG(kGraphExecutorLogVerboseLevel)
                  << "o_indptr (offset=" << o_indptr_offset
                  << ", size=" << num_o_indptr << "): " << o_indptr;
            }
          }
        }
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "window_size_left: " << sliding_window_;
      VLOG(kGraphExecutorLogVerboseLevel) << "scale: " << scale_;
    }

    xllm::kernel::batch_decode(attention_params);
    if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
      VLOG(kGraphExecutorLogVerboseLevel) << "query: " << query;
      VLOG(kGraphExecutorLogVerboseLevel) << "output: " << output_tensor;
    }
  }

  output_tensor = output_tensor.view({-1, num_heads_ * head_size_});
  return {output_tensor, output_lse};
}

}  // namespace layer
}  // namespace xllm