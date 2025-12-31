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
#include "kernels/ops_api.h"
#include "kernels/cuda/cuda_ops_api.h"

DECLARE_bool(enable_chunked_prefill);

namespace {

void lse_combine(torch::Tensor shared_o, 
                 torch::Tensor shared_lse, 
                 torch::Tensor unshared_o, 
                 torch::Tensor unshared_lse, 
                 torch::Tensor output) {
  xllm::kernel::cuda::lse_combine(output, shared_o, shared_lse, unshared_o, unshared_lse);
}
} // namespace

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
                                               attn_metadata.shared_k_cache, 
                                               attn_metadata.shared_v_cache);
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
      attention_params.plan_info = attn_metadata.plan_info;
      attention_params.key = key;
      attention_params.value = value;
      // LOG(INFO) << "key.shape: " << key.sizes();
      // LOG(INFO) << "value.shape: " << value.sizes();
      xllm::kernel::batch_prefill(attention_params);
      
    }
  } else {
    LLM_NVTX_RANGE("attention_decode");
    // LOG(INFO) << "query: " << query;
    // LOG(FATAL) << "after batch_prefill.";
    if (FLAGS_max_decode_rounds > 0) {
      LLM_NVTX_RANGE("attention_decode_with_shared");
      
      auto fp32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      // query: [total_beam, num_heads, head_dim]
      // LOG(INFO) << "query.shape: " << query.sizes();
      // output: [total_beam, num_heads, head_dim]
      // LOG(INFO) << "output.shape: " << output.sizes();

      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;

      torch::Tensor shared_q;
      torch::Tensor shared_lse;
      torch::Tensor shared_o;
      int32_t group_size;
      
      {
        LLM_NVTX_RANGE_COLOR("prepare_shared_query", 0xFF808080);  // Gray
        // [batch_size, beam_size * num_heads, head_dim]
        shared_q = query.clone();
        shared_q = shared_q.view({batch_size, beam_size, num_heads_, head_size_});
        group_size = num_heads_ / num_kv_heads_;
        
        shared_q = shared_q.view({batch_size, beam_size, num_kv_heads_, group_size, head_size_});
        // [batch_size, num_kv_heads_, beam_size, group_size, head_size_]
        shared_q = shared_q.permute({0, 2, 1, 3, 4}).contiguous();
        
        // [batch_size, num_kv_heads_ * beam_size * group_size, head_size_]
        shared_q = shared_q.view({batch_size, num_kv_heads_ * beam_size * group_size, head_size_});
        
        // 此时qk变成了 [beam_size * num_heads, head_dim] * [kv_seq_len, head_dim]
        // 防止了kv被load beam_size次，这里只需要load一次

        // shared
        shared_lse = 
          torch::zeros({shared_q.size(0), shared_q.size(1), 1}, fp32_options);
        shared_o = 
          torch::zeros_like(shared_q);
      }

      {
        LLM_NVTX_RANGE_COLOR("batch_prefill_shared", 0xFF00FF00);  // Green
        xllm::kernel::AttentionParams shared_attention_params;
        shared_attention_params.return_lse = true;
        shared_attention_params.query = shared_q;
        shared_attention_params.output = shared_o;
        shared_attention_params.output_lse = shared_lse;

        // shared_attention_params.max_seq_len = attn_metadata.max_seq_len;
        shared_attention_params.window_size_left = sliding_window_;
        shared_attention_params.scale = scale_;
        shared_attention_params.compute_dtype = attn_metadata.compute_dtype;
        // for flashinfer
        shared_attention_params.float_workspace_buffer =
            FlashinferWorkspace::get_instance().get_float_workspace_buffer();
        shared_attention_params.int_workspace_buffer =
            FlashinferWorkspace::get_instance().get_int_workspace_buffer();
        shared_attention_params.page_locked_int_workspace_buffer =
            FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();

        shared_attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
        shared_attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;

        // TODO: support chunked prefill
        CHECK(!attn_metadata.is_chunked_prefill)
            << "chunked prefill is not supported";
            
        shared_attention_params.key = attn_metadata.shared_k_cache;
        shared_attention_params.value = attn_metadata.shared_v_cache;

        shared_attention_params.plan_info = attn_metadata.plan_info;
        // shared_attention_params.is_decode_shared = true;

        xllm::kernel::batch_prefill(shared_attention_params);
      }
      
      {
        LLM_NVTX_RANGE_COLOR("reshape_shared_output", 0xFFFF00FF);  // Magenta
        // batch_prefill的输出是 [batch_size, num_kv_heads_ * beam_size * group_size, head_size_]
        // 需要reshape回 [batch_size, num_kv_heads_, beam_size, group_size, head_size_]
        shared_o = shared_o.view({batch_size, num_kv_heads_, beam_size, group_size, head_size_});
        // permute回 [batch_size, beam_size, num_kv_heads_, group_size, head_size_]
        shared_o = shared_o.permute({0, 2, 1, 3, 4}).contiguous();
        // 然后reshape回原始的 [batch_size * beam_size, num_heads_, head_size_]
        shared_o = shared_o.view({batch_size * beam_size, num_heads_, head_size_});
        
        // batch_prefill的输出是 [batch_size, num_kv_heads_ * beam_size * group_size, 1]
        // 需要reshape回 [batch_size, num_kv_heads_, beam_size, group_size, 1]
        shared_lse = shared_lse.view({batch_size, num_kv_heads_, beam_size, group_size, 1});
        // permute回 [batch_size, beam_size, num_kv_heads_, group_size, 1]
        shared_lse = shared_lse.permute({0, 2, 1, 3, 4}).contiguous();
        // 然后reshape回原始的 [batch_size * beam_size, num_heads_, 1]
        shared_lse = shared_lse.view({batch_size * beam_size, num_heads_, 1});
      }
      // unshared

      {
        LLM_NVTX_RANGE_COLOR("decoder_reshape_and_cache", 0xFF008080);  // Teal
        key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
        value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});
        
        xllm::kernel::cuda::decoder_reshape_and_cache(key, 
                                                      value, 
                                                      k_cache, 
                                                      v_cache, 
                                                      attn_metadata.block_table, 
                                                      attn_metadata.step);
      }
      
      torch::Tensor unshared_lse = attn_metadata.unshared_lse;
      torch::Tensor unshared_o = attn_metadata.unshared_o;
      
      {
        LLM_NVTX_RANGE_COLOR("batch_decode_unshared", 0xFFFF0000);  // Red
        xllm::kernel::AttentionParams unshared_attention_params;
        // auto unshared_lse = std::nullopt;
        
        unshared_attention_params.return_lse = true;
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
        
        // total_beams = batch_size * beam_size
        query = query.view({-1, 1, num_heads_, head_size_});
        unshared_o = unshared_o.view({-1, 1, num_heads_, head_size_});
        // LOG(INFO) << "query.shape: " << query.sizes();
        unshared_attention_params.query = query;
        unshared_attention_params.output = unshared_o;

        int64_t max_decode_step = k_cache.size(2);

        k_cache = k_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});
        v_cache = v_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});

        unshared_attention_params.k_cache = k_cache;
        unshared_attention_params.v_cache = v_cache;

        unshared_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
        unshared_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
        unshared_attention_params.paged_kv_last_page_len = attn_metadata.paged_kv_last_page_len;

        xllm::kernel::batch_decode(unshared_attention_params);
      }
      // LOG(INFO) << "after kernel::batch_decode.";
      // combine
      {
        LLM_NVTX_RANGE_COLOR("lse_combine", 0xFF0000FF);  // Blue
        unshared_o = unshared_o.view({-1, num_heads_, head_size_});
        // LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
        xllm::kernel::cuda::lse_combine(output, shared_o, shared_lse, unshared_o, unshared_lse);
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
        decode_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
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