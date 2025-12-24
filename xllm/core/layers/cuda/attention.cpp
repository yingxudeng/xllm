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

#include "flashinfer_workspace.h"
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
      sliding_window_(sliding_window - 1) {
        rec_kernel_ = std::make_unique<kernel::cuda::triton::RecTorchKernel>();
      }

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  LOG(INFO) << "inner AttentionImpl::forward.";
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();
  if (FLAGS_max_decode_rounds > 0) {
    if (attn_metadata.is_prefill) {
      // LOG(INFO) << "attn_metadata.shared_k_cache.shape: "
      //           << attn_metadata.shared_k_cache.sizes();
      // LOG(INFO) << "attn_metadata.shared_v_cache.shape: "
      //           << attn_metadata.shared_v_cache.sizes();
      // LOG(INFO) << "before prefill_reshape_and_cache.";
      // LOG(INFO) << "attn_metadata.shared_k_cache[0]: " << attn_metadata.shared_k_cache[0];
      rec_kernel_->prefill_reshape_and_cache(key, 
                                             value, 
                                             attn_metadata.shared_k_cache, 
                                             attn_metadata.shared_v_cache);
      // LOG(INFO) << "after prefill_reshape_and_cache.";
      // LOG(INFO) << "attn_metadata.shared_k_cache[0]: " << attn_metadata.shared_k_cache[0];
    }
  } else {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }
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
  LOG(INFO) << "attn_metadata.is_prefill: " << attn_metadata.is_prefill;
  // TODO: support chunked prefill
  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";
  if (attn_metadata.is_prefill) {
    attention_params.key = key;
    attention_params.value = value;
    LOG(INFO) << "key.shape: " << key.sizes();
    LOG(INFO) << "value.shape: " << value.sizes();
    xllm::kernel::batch_prefill(attention_params);
    
  } else {
    if (FLAGS_max_decode_rounds > 0) {
      // attention_params.print();
      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      LOG(INFO) << "batch_size: " << batch_size;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;
      LOG(INFO) << "beam_size: " << beam_size;

      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});

      
      LOG(INFO) << "attn_metadata.block_table: " << attn_metadata.block_table;
      LOG(INFO) << "attn_metadata.kv_seq_lens: " << attn_metadata.kv_seq_lens;
      LOG(INFO) << "attn_metadata.step: " << attn_metadata.step;

      LOG(INFO) << "k_cache.shape: " << k_cache.sizes();
      LOG(INFO) << "v_cache.shape: " << v_cache.sizes();

      LOG(INFO) << "query.shape: " << query.sizes();
      LOG(INFO) << "output.shape: " << output.sizes();
      LOG(INFO) << "key.shape: " << key.sizes();
      LOG(INFO) << "value.shape: " << value.sizes();
      
      rec_kernel_->decoder_reshape_and_cache(key, 
                                             value, 
                                             k_cache, 
                                             v_cache, 
                                             attn_metadata.block_table, 
                                             attn_metadata.step);

      query = query.view({batch_size, beam_size, num_heads_, head_size_});
      output = output.view({batch_size, beam_size, num_heads_, head_size_});

      // LOG(INFO) << "query.shape: " << query.sizes();
      // LOG(INFO) << "output.shape: " << output.sizes();

      LOG(INFO) << "key: " << key;
      LOG(INFO) << "value: " << value;
      LOG(INFO) << "query: " << query;
      
      output = rec_kernel_->xattention(query, 
                              attn_metadata.shared_k_cache, 
                              attn_metadata.shared_v_cache, 
                              k_cache, 
                              v_cache, 
                              attn_metadata.kv_seq_lens, 
                              attn_metadata.block_table, 
                              scale_, 
                              attn_metadata.step);
      LOG(INFO) << "output: " << output;
      // LOG(FATAL) << "after xattention.";

    } else {
      LOG(INFO) << "key: " << key;
      LOG(INFO) << "value: " << value;
      query = query.view({-1, 1, num_heads_, head_size_});
      output = output.view({-1, 1, num_heads_, head_size_});

      attention_params.query = query;
      attention_params.output = output;
      attention_params.k_cache = k_cache;
      attention_params.v_cache = v_cache;

      // LOG(INFO) << "k_cache.shape: " << k_cache.sizes();
      // LOG(INFO) << "v_cache.shape: " << v_cache.sizes();

      // LOG(INFO) << "query.shape: " << query.sizes();
      // LOG(INFO) << "key.shape: " << key.sizes();
      // LOG(INFO) << "value.shape: " << value.sizes();
      // LOG(INFO) << "output.shape: " << output.sizes();

      // for flashinfer
      attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
      attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      attention_params.paged_kv_last_page_len =
          attn_metadata.paged_kv_last_page_len;
      LOG(INFO) << "before batch_decode.";
      
      attention_params.print();
      LOG(INFO) << "query: " << query;
      xllm::kernel::batch_decode(attention_params);
      
      LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
    }
    
    
    
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm