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

namespace {
torch::Tensor select_target_kv_cache(torch::Tensor kv_cache, 
                                     torch::Tensor block_table) {
  torch::Tensor indices = block_table.flatten().to(torch::kLong);
  LOG(INFO) << "indices: " << indices;
  torch::Tensor selected_cache = kv_cache.index_select(0, indices);
  LOG(INFO) << "selected_cache.shape: " << selected_cache.sizes();
  return selected_cache;
}

torch::Tensor lse_combine(torch::Tensor shared_o, 
                          torch::Tensor shared_lse, 
                          torch::Tensor unshared_o, 
                          torch::Tensor unshared_lse) {
  // 1. 计算 element-wise 最大 LSE
    // [batch, num_heads, 1]
    torch::Tensor li_max = torch::max(shared_lse, unshared_lse);

    // 2. 计算以 2 为底的指数差
    // [batch, num_heads, 1]
    torch::Tensor exp_li = torch::exp2(shared_lse - li_max);
    torch::Tensor exp_lij = torch::exp2(unshared_lse - li_max);

    // 3. 计算合并后的新 LSE
    // [batch, num_heads, 1]
    torch::Tensor li_new = li_max + torch::log2(exp_li + exp_lij);

    // 4. 计算归一化权重
    // 此时 lse 和 li_new 都是 [B, H, 1]，相减也是 [B, H, 1]
    // 这里的形状可以直接与 o [B, H, D] 进行广播，无需 unsqueeze
    torch::Tensor wi = torch::exp2(shared_lse - li_new);
    torch::Tensor wij = torch::exp2(unshared_lse - li_new);

    // 5. 加权合并输出 (自动广播: [B,H,1] * [B,H,D] -> [B,H,D])
    torch::Tensor o_online = wi * shared_o + wij * unshared_o;

  return o_online;
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
      auto fp32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      // query: [total_beam, num_heads, head_dim]
      LOG(INFO) << "query.shape: " << query.sizes();
      LOG(INFO) << "output.shape: " << output.sizes();

      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      LOG(INFO) << "batch_size: " << batch_size;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;
      LOG(INFO) << "beam_size: " << beam_size;

      query = query.view({batch_size, beam_size, num_heads_, head_size_});
      int32_t group_size = num_heads_ / num_kv_heads_;
      query = query.view({batch_size, beam_size, num_kv_heads_, group_size, head_size_});
      // [batch_size, num_kv_heads_, beam_size, group_size, head_size_]
      query = query.permute({0, 2, 1, 3, 4}).contiguous();

      // 相当于q_seq_len从1变成了beam_size，并且确保了显存布局正确
      // [batch_size, beam_size * num_heads, head_dim]
      query = query.view({batch_size, num_kv_heads_ * beam_size * group_size, head_size_});

      // 此时qk变成了 [beam_size * num_heads, head_dim] * [kv_seq_len, head_dim]
      // 防止了kv被load beam_size次，这里只需要load一次

      // LOG(INFO) << "key: " << key;
      // LOG(INFO) << "value: " << value;
      // LOG(INFO) << "query: " << query;

      // shared
      // auto shared_lse = std::nullopt;
      torch::Tensor shared_lse = 
        torch::zeros({query.size(0), query.size(1), 1}, fp32_options);
      torch::Tensor shared_o = 
        torch::zeros_like(output);
      LOG(INFO) << "shared_lse.shape: " << shared_lse.sizes();
      xllm::kernel::AttentionParams shared_attention_params;
      shared_attention_params.return_lse = true;
      shared_attention_params.query = query;
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
      // 这里相当于q_seq_len从1变成了beam_size，又因为可能有batch个请求，所以应该是
      // arange(0, batch_size, beam_size)
      shared_attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
      LOG(INFO) << "shared_attention_params.kv_cu_seq_lens: "
                << *shared_attention_params.kv_cu_seq_lens;
      LOG(INFO) << "shared_attention_params.q_cu_seq_lens: "
                << *shared_attention_params.q_cu_seq_lens;

      // TODO: support chunked prefill
      CHECK(!attn_metadata.is_chunked_prefill)
          << "chunked prefill is not supported";
          
      shared_attention_params.key = attn_metadata.shared_k_cache;
      shared_attention_params.value = attn_metadata.shared_v_cache;

      LOG(INFO) << "attn_metadata.shared_k_cache.shape: " 
                << attn_metadata.shared_k_cache.sizes();
      LOG(INFO) << "attn_metadata.shared_v_cache.shape: " 
                << attn_metadata.shared_v_cache.sizes();

      xllm::kernel::batch_prefill(shared_attention_params);
      LOG(INFO) << "shared_lse: " << shared_lse;
      // LOG(FATAL) << "after xllm::kernel::batch_prefill(shared_attention_params).";
      // unshared

      

      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});
      LOG(INFO) << "key.shape: " << key.sizes();
      LOG(INFO) << "value.shape: " << value.sizes();
      
      LOG(INFO) << "attn_metadata.block_table: " << attn_metadata.block_table;
      // LOG(INFO) << "attn_metadata.kv_seq_lens: " << attn_metadata.kv_seq_lens;
      LOG(INFO) << "attn_metadata.step: " << attn_metadata.step;

      // LOG(INFO) << "k_cache.shape: " << k_cache.sizes();
      // LOG(INFO) << "v_cache.shape: " << v_cache.sizes();

      LOG(INFO) << "query.shape: " << query.sizes();
      LOG(INFO) << "shared_o.shape: " << shared_o.sizes();
      
      
      rec_kernel_->decoder_reshape_and_cache(key, 
                                             value, 
                                             k_cache, 
                                             v_cache, 
                                             attn_metadata.block_table, 
                                             attn_metadata.step);
      LOG(INFO) << "begin prepare for unshared.";
      torch::Tensor unshared_lse = 
        torch::zeros({query.size(0), query.size(1), 1}, fp32_options);
      LOG(INFO) << "unshared_lse.shape: " << shared_lse.sizes();
      torch::Tensor unshared_o = 
        torch::zeros_like(output);
      xllm::kernel::AttentionParams unshared_attention_params;
      // auto unshared_lse = std::nullopt;
      
      unshared_attention_params.return_lse = true;
      unshared_attention_params.output_lse = unshared_lse;
      // unshared_attention_params.max_seq_len = attn_metadata.max_seq_len;
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

      // decode可能不需要cu_seq_lens
      // unshared_attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
      // unshared_attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
      LOG(INFO) << "attn_metadata.is_prefill: " << attn_metadata.is_prefill;
      // TODO: support chunked prefill
      CHECK(!attn_metadata.is_chunked_prefill)
          << "chunked prefill is not supported";
      
      // total_beams = batch_size * beam_size
      query = query.view({-1, 1, num_heads_, head_size_});
      unshared_o = unshared_o.view({-1, 1, num_heads_, head_size_});
      LOG(INFO) << "query.shape: " << query.sizes();
      unshared_attention_params.query = query;
      unshared_attention_params.output = unshared_o;
      // k_cache: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
      // 需要把kv_cache的第0维度变成batch_size，先用block_table过滤一遍
      

      unshared_attention_params.k_cache = 
        select_target_kv_cache(k_cache, attn_metadata.block_table);
      unshared_attention_params.v_cache = 
        select_target_kv_cache(v_cache, attn_metadata.block_table);
      // [batch_size, beam_size, max_decode_step, kv_heads, head_dim]
      
      int64_t max_decode_step = unshared_attention_params.k_cache.size(2);

      unshared_attention_params.k_cache = 
        unshared_attention_params.k_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});
      unshared_attention_params.v_cache = 
        (*(unshared_attention_params.v_cache)).view({-1, max_decode_step, num_kv_heads_, head_size_});
      LOG(INFO) << "unshared_attention_params.k_cache.shape: "
                << unshared_attention_params.k_cache.sizes();
      LOG(INFO) << "unshared_attention_params.v_cache.shape: "
                << (*(unshared_attention_params.v_cache)).sizes();
      
      // 这个应该是batch_size * beam_size粒度的，当作并行轴来处理
      // 实际的计算是NHD，既[1, num_heads, head_dim] * [max_decode_step, kv_heads, head_dim]
      // 这个是其实是拍平的block_table值，所以为arange(0, batch_size * beam_size)
      // unshared_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      unshared_attention_params.paged_kv_indices = 
        torch::arange(total_beam, attn_metadata.paged_kv_indices.options());
      // 这个属性代表的应该是block_table的累加值，所以为arange(0, batch_size * beam_size + 1)
      // unshared_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
      unshared_attention_params.paged_kv_indptr =
        torch::arange(total_beam + 1, attn_metadata.paged_kv_indptr.options());
      // 这个属性代表的是尾块的seq_ken，当前一个block相当于有step + 1个有效值，所以这个应该是batch_size * beam_size 个step + 1
      // unshared_attention_params.paged_kv_last_page_len =
      //     attn_metadata.paged_kv_last_page_len;
      unshared_attention_params.paged_kv_last_page_len =
          torch::full({total_beam}, attn_metadata.step + 1, attn_metadata.paged_kv_last_page_len.options());
      LOG(INFO) << "unshared_attention_params.paged_kv_indptr: " 
                << unshared_attention_params.paged_kv_indptr;
      LOG(INFO) << "unshared_attention_params.paged_kv_indices: " 
                << unshared_attention_params.paged_kv_indices;
      LOG(INFO) << "unshared_attention_params.paged_kv_last_page_len: " 
                << unshared_attention_params.paged_kv_last_page_len;
      // LOG(FATAL) << "before batch_decode.";
      xllm::kernel::batch_decode(unshared_attention_params);
      // combine
      unshared_o = unshared_o.view({-1, num_heads_, head_size_});
      LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      
      auto final_out = lse_combine(shared_o, shared_lse, 
                                   unshared_o, unshared_lse);

      // LOG(INFO) << "output: " << output;
      LOG(INFO) << "unshared_lse: " << unshared_lse;
      LOG(INFO) << "final_out: " << final_out;
      LOG(FATAL) << "after xattention.";

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
      LOG(INFO) << "attention_params.paged_kv_indptr: " 
                << attention_params.paged_kv_indptr;
      LOG(INFO) << "attention_params.paged_kv_indices: " 
                << attention_params.paged_kv_indices;
      LOG(INFO) << "attention_params.paged_kv_last_page_len: " 
                << attention_params.paged_kv_last_page_len;
      LOG(INFO) << "attention_params.kv_cu_seq_lens: "
                << *(attention_params.kv_cu_seq_lens);
      LOG(INFO) << "attention_params.q_cu_seq_lens: "
                << *(attention_params.q_cu_seq_lens);
      LOG(INFO) << "before batch_decode.";
      
      attention_params.print();
      // LOG(INFO) << "query: " << query;
      xllm::kernel::batch_decode(attention_params);
      
      // LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
    }
    
    
    
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm