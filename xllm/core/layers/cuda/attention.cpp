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

#include "common/nvtx_helper.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);

namespace {

void lse_combine(torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse,
                 torch::Tensor output) {
  xllm::kernel::cuda::lse_combine(
      output, shared_o, shared_lse, unshared_o, unshared_lse);
}
}  // namespace

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
      sliding_window_(sliding_window - 1) {
  rec_kernel_ = std::make_unique<kernel::cuda::triton::RecTorchKernel>();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  LLM_NVTX_RANGE("AttentionImpl_forward");

  // LOG(INFO) << "inner AttentionImpl::forward.";
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  {
    LLM_NVTX_RANGE_COLOR("attention_reshape_inputs", 0xFF808080);  // Gray
    query = query.view({-1, num_heads_, head_size_});
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});
    output = output.view({-1, num_heads_, head_size_});
  }

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  if (FLAGS_max_decode_rounds == 0) {
    {
      LLM_NVTX_RANGE_COLOR("reshape_paged_cache", 0xFF800080);  // Purple
      xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
      reshape_paged_cache_params.key = key;
      reshape_paged_cache_params.value = value;
      reshape_paged_cache_params.k_cache = k_cache;
      reshape_paged_cache_params.v_cache = v_cache;
      reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
      xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
    }
  }

  if (attn_metadata.is_prefill) {
    LLM_NVTX_RANGE("attention_prefill");

    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";
    if (FLAGS_max_decode_rounds > 0) {
      {
        LLM_NVTX_RANGE_COLOR("prefill_reshape_and_cache", 0xFF008080);  // Teal
        rec_kernel_->prefill_reshape_and_cache(key, 
                                               value, 
                                               attn_metadata.full_k_cache, 
                                               attn_metadata.full_v_cache);
      }
    }

    {
      LLM_NVTX_RANGE_COLOR("batch_prefill", 0xFF00FF00);  // Green
      xllm::kernel::AttentionParams attention_params;
      attention_params.query = query;
      attention_params.output = output;
      attention_params.output_lse = output_lse;
      // attention_params.max_seq_len = attn_metadata.max_seq_len;
      attention_params.window_size_left = sliding_window_;
      attention_params.scale = scale_;
      attention_params.compute_dtype = attn_metadata.compute_dtype;
      // for flashinfer
      attention_params.float_workspace_buffer =
          FlashinferWorkspace::get_instance().get_float_workspace_buffer();
      attention_params.int_workspace_buffer =
          FlashinferWorkspace::get_instance().get_int_workspace_buffer();
      attention_params.page_locked_int_workspace_buffer =
          FlashinferWorkspace::get_instance()
              .get_page_locked_int_workspace_buffer();
      attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
      attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
      // LOG(INFO) << "attn_metadata.is_prefill: " << attn_metadata.is_prefill;
      // TODO: support chunked prefill
      attention_params.plan_info = attn_metadata.prefill_plan_info;
      attention_params.key = key;
      attention_params.value = value;
      // LOG(INFO) << "key.shape: " << key.sizes();
      // LOG(INFO) << "value.shape: " << value.sizes();
      xllm::kernel::batch_prefill(attention_params);
    }
  } else {
    LLM_NVTX_RANGE("attention_decode");

    if (FLAGS_max_decode_rounds > 0) {
      // LOG(INFO) << "attention_decode_with_shared";
      LLM_NVTX_RANGE("attention_decode_with_shared");

      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;

      // [max_shared_kv_len, num_kv_heads_, head_size_]
      torch::Tensor full_k_cache = attn_metadata.full_k_cache;
      torch::Tensor full_v_cache = attn_metadata.full_v_cache;

      // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_})
                .contiguous();
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_})
                  .contiguous();
      int32_t full_kv_len = full_k_cache.size(0);
      int32_t unshared_offset = batch_size * FLAGS_max_token_per_req;
      int32_t max_decode_step = k_cache.size(2);
      // LOG(INFO) << "full_kv_len: " << full_kv_len;
      // LOG(INFO) << "unshared_offset: " << unshared_offset;
      auto unshared_k_cache = full_k_cache.slice(0, unshared_offset, full_kv_len);
      auto unshared_v_cache = full_v_cache.slice(0, unshared_offset, full_kv_len);
      // LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
      // LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
      unshared_k_cache = unshared_k_cache.view({batch_size, beam_size, max_decode_step, num_kv_heads_, head_size_});
      unshared_v_cache = unshared_v_cache.view({batch_size, beam_size, max_decode_step, num_kv_heads_, head_size_});


      xllm::kernel::cuda::decoder_reshape_and_cache(key,
                                                    value,
                                                    unshared_k_cache,
                                                    unshared_v_cache,
                                                    attn_metadata.naive_block_table,
                                                    attn_metadata.step);

      full_k_cache = full_k_cache.unsqueeze(1);
      full_v_cache = full_v_cache.unsqueeze(1);


      {
        LLM_NVTX_RANGE_COLOR("batch_decode_unshared", 0xFFFF0000);  // Red
        xllm::kernel::AttentionParams unshared_attention_params;
        auto unshared_lse = std::nullopt;

        unshared_attention_params.return_lse = false;
        unshared_attention_params.output_lse = unshared_lse;

        unshared_attention_params.window_size_left = sliding_window_;
        unshared_attention_params.scale = scale_;
        unshared_attention_params.compute_dtype = attn_metadata.compute_dtype;
        // for flashinfer
        unshared_attention_params.float_workspace_buffer =
            FlashinferWorkspace::get_instance().get_float_workspace_buffer();
        unshared_attention_params.int_workspace_buffer =
            FlashinferWorkspace::get_instance().get_int_workspace_buffer();
        unshared_attention_params.page_locked_int_workspace_buffer =
            FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();

        // TODO: support chunked prefill
        CHECK(!attn_metadata.is_chunked_prefill)
            << "chunked prefill is not supported";

        unshared_attention_params.query = query;
        unshared_attention_params.output = output;

        unshared_attention_params.k_cache = full_k_cache;
        unshared_attention_params.v_cache = full_v_cache;

        unshared_attention_params.paged_kv_indices =
            attn_metadata.decode_paged_kv_indices;
        unshared_attention_params.paged_kv_indptr =
            attn_metadata.decode_paged_kv_indptr;
        unshared_attention_params.paged_kv_last_page_len =
            attn_metadata.decode_paged_kv_last_page_len;

        unshared_attention_params.plan_info = attn_metadata.decode_plan_info;

        xllm::kernel::batch_decode(unshared_attention_params);
        // LOG(INFO) << "output: " << o;
        // LOG(FATAL) << "after batch_decode.";
      }

      // LOG(INFO) << "output: " << output;
      // LOG(FATAL) << "after batch_decode.";
    } else {
      LLM_NVTX_RANGE("attention_decode_standard");

      {
        LLM_NVTX_RANGE_COLOR("batch_decode_standard", 0xFF0000FF);  // Blue
        query = query.view({-1, 1, num_heads_, head_size_});
        output = output.view({-1, 1, num_heads_, head_size_});
        xllm::kernel::AttentionParams decode_attention_params;
        decode_attention_params.window_size_left = sliding_window_;
        decode_attention_params.scale = scale_;
        decode_attention_params.compute_dtype = attn_metadata.compute_dtype;

        decode_attention_params.query = query;
        decode_attention_params.output = output;
        decode_attention_params.k_cache = k_cache;
        decode_attention_params.v_cache = v_cache;

        // for flashinfer
        decode_attention_params.float_workspace_buffer =
            FlashinferWorkspace::get_instance().get_float_workspace_buffer();
        decode_attention_params.int_workspace_buffer =
            FlashinferWorkspace::get_instance().get_int_workspace_buffer();
        decode_attention_params.page_locked_int_workspace_buffer =
            FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();
        decode_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
        decode_attention_params.paged_kv_indices =
            attn_metadata.paged_kv_indices;
        decode_attention_params.paged_kv_last_page_len =
            attn_metadata.paged_kv_last_page_len;

        xllm::kernel::batch_decode(decode_attention_params);
      }

      LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
    }
  }
  // LOG(INFO) << "output: " << output;
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm