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
                                               attn_metadata.shared_v_cache,
                                               attn_metadata.kv_cu_seq_lens);
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

      LOG(INFO) << "query.shape: " << query.sizes();
      LOG(INFO) << "output.shape: " << output.sizes();
      LOG(INFO) << "k_cache.shape: " << k_cache.sizes();
      LOG(INFO) << "v_cache.shape: " << v_cache.sizes();
      LOG(INFO) << "attn_metadata.paged_kv_indptr: " << attn_metadata.paged_kv_indptr;
      LOG(INFO) << "attn_metadata.paged_kv_indices: " << attn_metadata.paged_kv_indices;
      LOG(INFO) << "attn_metadata.paged_kv_last_page_len: " << attn_metadata.paged_kv_last_page_len;
      LOG(INFO) << "attn_metadata.kv_cu_seq_lens: " << attn_metadata.kv_cu_seq_lens;
      LOG(INFO) << "attn_metadata.q_cu_seq_lens: " << attn_metadata.q_cu_seq_lens;
      LOG(INFO) << "attn_metadata.shared_k_cache.shape: " << attn_metadata.shared_k_cache.sizes();
      LOG(INFO) << "attn_metadata.shared_v_cache.shape: " << attn_metadata.shared_v_cache.sizes();
      LOG(INFO) << "attn_metadata.unshared_lse.shape: " << attn_metadata.unshared_lse.sizes();
      LOG(INFO) << "attn_metadata.unshared_o.shape: " << attn_metadata.unshared_o.sizes();
      LOG(INFO) << "attn_metadata.block_table: " << attn_metadata.block_table;
      // [max_shared_kv_len, num_kv_heads_, head_size_]
      torch::Tensor shared_k_cache = attn_metadata.shared_k_cache;
      torch::Tensor shared_v_cache = attn_metadata.shared_v_cache;
      uint32_t shared_kv_len = shared_k_cache.size(0) / batch_size;
      
      uint32_t max_decode_step = k_cache.size(2);
      // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
      torch::Tensor unshared_k_cache = k_cache.view({-1, num_kv_heads_, head_size_});
      torch::Tensor unshared_v_cache = v_cache.view({-1, num_kv_heads_, head_size_});
      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});

      rec_kernel_->decoder_reshape_and_cache(key, 
                                             value, 
                                             k_cache, 
                                             v_cache,
                                             attn_metadata.block_table,
                                             attn_metadata.step);
      
      LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
      LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
      LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
      LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
      
      // [batch_size * shared_kv_len + batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
      auto full_k_cache = torch::cat({shared_k_cache, unshared_k_cache}, 0);
      full_k_cache = full_k_cache.unsqueeze(1);
      LOG(INFO) << "full_k_cache.shape: " << full_k_cache.sizes();
      auto full_v_cache = torch::cat({shared_v_cache, unshared_v_cache}, 0);
      full_v_cache = full_v_cache.unsqueeze(1);
      LOG(INFO) << "full_v_cache.shape: " << full_v_cache.sizes();

      // 代表每个batch的prompt kv 长度
      auto batch_shared_kv_lens = torch::diff(attn_metadata.kv_cu_seq_lens);
      LOG(INFO) << "batch_shared_kv_lens: " << batch_shared_kv_lens;

      // [beam_size, shared_kv_len]
      // 用来做mask，表示每一个batch的长度，本来是一维的，比如[3, 5]
      // 现在是[[3, 3], [5, 5]]，用来做掩码
      auto beam_shared_kv_expanded = 
        batch_shared_kv_lens.unsqueeze(1).expand({-1, shared_kv_len});
      
      // 这个表示shared_kv_len上的偏移量，用arange表示，后续会取mask后的值，作为真实索引
      auto shared_kv_len_offsets = torch::arange(0, shared_kv_len, attn_metadata.paged_kv_indices.options());
      shared_kv_len_offsets = shared_kv_len_offsets.unsqueeze(0).expand({batch_size, -1});
      LOG(INFO) << "shared_kv_len_offsets: " << shared_kv_len_offsets;
      // 相当于[[0, 1, 2, 3],     <    [2, 2, 2, 2]
      //       [0, 1, 2, 3]]          [3, 3, 3, 3]
      // 这样就等于[[True, True, True, False],
      //             [True, True, True, True]]
      auto mask = shared_kv_len_offsets < beam_shared_kv_expanded;
      LOG(INFO) << "mask: " << mask;

      auto batch_offsets = torch::arange(0, batch_size, attn_metadata.paged_kv_indices.options());
      // [batch_size, shared_kv_len]
      // 这里变成了[[0, 0, 0, 0],
      //             [1, 1, 1, 1]]
      auto shared_batch_offsets = batch_offsets.unsqueeze(1).expand({-1, shared_kv_len});
      // 这里变成了[[0, 0, 0, 0],
      //             [4, 4, 4, 4]]
      shared_batch_offsets = shared_batch_offsets * shared_kv_len;
      LOG(INFO) << "shared_batch_offsets: " << shared_batch_offsets;

      // 这里变成了[[0, 1, 2, 3],
      //             [4, 5, 6, 7]]
      auto shared_kv_indices = shared_batch_offsets + shared_kv_len_offsets;
      // 这里变成了[[0, 1, 2, 0],
      //             [4, 5, 6, 7]]
      // 如果用masked_select，实际上会变成[0, 1, 2, 4, 5, 6, 7]，
      // 但是我们这里还没有考虑unshared的kv部分，需要考虑，然后做concat
      // 比如unshared部分为 [[100, 101], 
      //                   [102, 103]]，那么最终的kv_indices应该是[0, 1, 2, 0 ,100, 101,]
      //                                                      [4, 5, 6, 7, 102, 103]   
      // 然后再想办法取[0, 1, 2, 100, 101, 4, 5, 6, 102, 103]     
      // shared_kv_indices = shared_kv_indices.masked_select(mask);
      shared_kv_indices = shared_kv_indices.masked_fill(~mask, 0);
      // 将 mask 中 False 的位置置为 0
      // 这里变成了[[0, 1, 2, 0],
      //          [4, 5, 6, 7]]
      LOG(INFO) << "shared_kv_indices: " << shared_kv_indices;
      
      // 而kv_cu_seq_len实际上就是paged_kv_indptr，因为前者是token粒度的，后者当block_size为1时，也是token粒度的
      // 但是还需要做一些处理，因为kv_cu_seq_len是engine侧产生的，因此只有prompt的长度，
      // 这个可能需要根据decode的步数，来为做扩展

      uint32_t unshared_begin_index = shared_kv_len * batch_size;
      LOG(INFO) << "unshared_begin_index: " << unshared_begin_index;
      // 这个batch_ids不能是这样的，应该从block_table中获取
      auto batch_ids = attn_metadata.block_table.select(1, 0);
      // auto batch_ids = torch::arange(0, batch_size, attn_metadata.paged_kv_indices.options());
      // [batch_size, beam_size, max_decode_step]
      batch_ids = batch_ids.unsqueeze(1).expand({-1, beam_size}).unsqueeze(2).expand({-1, -1, max_decode_step});
      
      batch_ids = batch_ids * beam_size * max_decode_step;
      auto beams_ids = torch::arange(0, beam_size, attn_metadata.paged_kv_indices.options());
      beams_ids = beams_ids.unsqueeze(0).expand({batch_size, -1}).unsqueeze(2).expand({-1, -1, max_decode_step});
      beams_ids = beams_ids * max_decode_step;
      auto max_decode_step_ids = torch::arange(0, max_decode_step, attn_metadata.paged_kv_indices.options());
      max_decode_step_ids = max_decode_step_ids.unsqueeze(0).expand({batch_size, -1}).unsqueeze(1).expand({-1, beam_size, -1});
      max_decode_step_ids = max_decode_step_ids * 1;
      
      LOG(INFO) << "batch_ids: " << batch_ids;
      LOG(INFO) << "beams_ids: " << beams_ids;
      LOG(INFO) << "max_decode_step_ids: " << max_decode_step_ids;
      
      auto unshared_kv_offsets = batch_ids + beams_ids + max_decode_step_ids;
      LOG(INFO) << "unshared_kv_offsets: " << unshared_kv_offsets;
      // [batch_size, beam_size, max_decode_step]
      auto unshared_kv_indices = unshared_kv_offsets + unshared_begin_index;
      LOG(INFO) << "unshared_kv_indices: " << unshared_kv_indices;
      // last_page_len实际上就是1了，因为block_size为1

      // [batch_size, beam_size, shared_kv_len]
      shared_kv_indices = shared_kv_indices.unsqueeze(1).expand({-1, beam_size, -1});
      LOG(INFO) << "shared_kv_indices_expanded: " << shared_kv_indices;
      auto full_kv_indices = torch::cat({shared_kv_indices, unshared_kv_indices}, 2);
      LOG(INFO) << "full_kv_indices: " << full_kv_indices;

      auto shared_mask = mask.unsqueeze(1).expand({-1, beam_size, -1});
      LOG(INFO) << "shared_mask: " << shared_mask;
      // auto unshared_mask = torch::ones_like(unshared_kv_indices, torch::kBool);
      // auto unshared_mask = torch::arange(0, max_decode_step, attn_metadata.paged_kv_indices.options());
      uint32_t current_step = attn_metadata.step;
      LOG(INFO) << "current_step: " << current_step;
      auto unshared_mask = max_decode_step_ids <= current_step;

      LOG(INFO) << "unshared_mask: " << unshared_mask;
      // [batch_size, beam_size, shared_kv_len + max_decode_step]
      auto full_mask = torch::cat({shared_mask, unshared_mask}, 2);
      LOG(INFO) << "full_mask: " << full_mask;

      auto paged_kv_indices = full_kv_indices.masked_select(full_mask);
      LOG(INFO) << "full_kv_indices_masked: " << paged_kv_indices;
      // 它其实就是paged_kv_indices
      
      // 接下来算indptr
      // [batch_size, beam_size]
      auto batch_beam_shared_kv_lens = batch_shared_kv_lens.unsqueeze(1).expand({-1, beam_size});
      // LOG(INFO) << "batch_beam_shared_kv_lens: " << batch_beam_shared_kv_lens;
      // shared_kv_Len + unshared_kv_Len
      uint32_t unshared_kv_len = current_step + 1;
      batch_beam_shared_kv_lens = batch_beam_shared_kv_lens + unshared_kv_len;
      // LOG(INFO) << "batch_beam_shared_kv_lens_plus_max_decode_step: " << batch_beam_shared_kv_lens;

      // 展平并计算累积和
      // [batch_size * beam_size]
      auto flattened = batch_beam_shared_kv_lens.flatten();
      auto cumsum_result = torch::cumsum(flattened, 0).to(batch_beam_shared_kv_lens.dtype());
      // LOG(INFO) << "cumsum_result: " << cumsum_result;
      // 在前面加 0
      auto paged_kv_indptr = torch::cat({
          torch::zeros({1}, cumsum_result.options()),
          cumsum_result
      }, 0);

      auto paged_kv_last_page_len = torch::ones({batch_size * beam_size}, attn_metadata.paged_kv_indices.options());
      // LOG(INFO) << "cumulative_indptr: " << cumulative_indptr;
      // input传过来的paged_kv_indices是以batch为粒度的，因此它的shape是[batch_size]
      // 
      // [batch_size]
      
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
        
        // total_beams = batch_size * beam_size
        // query = query.view({-1, 1, num_heads_, head_size_});
        // unshared_o = unshared_o.view({-1, 1, num_heads_, head_size_});
        // LOG(INFO) << "query.shape: " << query.sizes();
        auto o = torch::zeros_like(query);
        unshared_attention_params.query = query;
        unshared_attention_params.output = o;
        LOG(INFO) << "query.shape: " << query.sizes();
        LOG(INFO) << "unshared_o.shape: " << o.sizes();

        int64_t max_decode_step = k_cache.size(2);

        unshared_attention_params.k_cache = full_k_cache;
        unshared_attention_params.v_cache = full_v_cache;
        LOG(INFO) << "unshared_attention_params.k_cache: " << unshared_attention_params.k_cache.sizes();
        // 这个还不能用来做kv_indices，因为它没有考虑block_table
        unshared_attention_params.paged_kv_indices = paged_kv_indices;
        unshared_attention_params.paged_kv_indptr = paged_kv_indptr;
        unshared_attention_params.paged_kv_last_page_len = paged_kv_last_page_len;
        LOG(INFO) << "unshared_attention_params.paged_kv_last_page_len: " << unshared_attention_params.paged_kv_last_page_len;
        LOG(INFO) << "unshared_attention_params.paged_kv_indptr: " << unshared_attention_params.paged_kv_indptr;
        LOG(INFO) << "unshared_attention_params.paged_kv_indices: " << unshared_attention_params.paged_kv_indices;

        xllm::kernel::batch_decode(unshared_attention_params);
        LOG(INFO) << "output: " << o;
        LOG(FATAL) << "after batch_decode.";
      }
      
      

      // torch::Tensor shared_q;
      // torch::Tensor shared_lse;
      // torch::Tensor shared_o;
      // int32_t group_size;
      
      // {
      //   LLM_NVTX_RANGE_COLOR("prepare_shared_query", 0xFF808080);  // Gray
      //   // [batch_size, beam_size * num_heads, head_dim]
      //   shared_q = query.clone();
      //   shared_q = shared_q.view({batch_size, beam_size, num_heads_, head_size_});
      //   group_size = num_heads_ / num_kv_heads_;
        
      //   shared_q = shared_q.view({batch_size, beam_size, num_kv_heads_, group_size, head_size_});
      //   // [batch_size, num_kv_heads_, beam_size, group_size, head_size_]
      //   shared_q = shared_q.permute({0, 2, 1, 3, 4}).contiguous();
        
      //   // [batch_size, num_kv_heads_ * beam_size * group_size, head_size_]
      //   shared_q = shared_q.view({batch_size, num_kv_heads_ * beam_size * group_size, head_size_});
        
      //   // 此时qk变成了 [beam_size * num_heads, head_dim] * [kv_seq_len, head_dim]
      //   // 防止了kv被load beam_size次，这里只需要load一次

      //   // shared
      //   shared_lse = 
      //     torch::zeros({shared_q.size(0), shared_q.size(1), 1}, fp32_options);
      //   shared_o = 
      //     torch::zeros_like(shared_q);
      // }

      // {
      //   LLM_NVTX_RANGE_COLOR("batch_prefill_shared", 0xFF00FF00);  // Green
      //   xllm::kernel::AttentionParams shared_attention_params;
      //   shared_attention_params.return_lse = true;
      //   shared_attention_params.query = shared_q;
      //   shared_attention_params.output = shared_o;
      //   shared_attention_params.output_lse = shared_lse;

      //   // shared_attention_params.max_seq_len = attn_metadata.max_seq_len;
      //   shared_attention_params.window_size_left = sliding_window_;
      //   shared_attention_params.scale = scale_;
      //   shared_attention_params.compute_dtype = attn_metadata.compute_dtype;
      //   // for flashinfer
      //   shared_attention_params.float_workspace_buffer =
      //       FlashinferWorkspace::get_instance().get_float_workspace_buffer();
      //   shared_attention_params.int_workspace_buffer =
      //       FlashinferWorkspace::get_instance().get_int_workspace_buffer();
      //   shared_attention_params.page_locked_int_workspace_buffer =
      //       FlashinferWorkspace::get_instance()
      //           .get_page_locked_int_workspace_buffer();

      //   shared_attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
      //   shared_attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;

      //   // TODO: support chunked prefill
      //   CHECK(!attn_metadata.is_chunked_prefill)
      //       << "chunked prefill is not supported";
            
      //   shared_attention_params.key = attn_metadata.shared_k_cache;
      //   shared_attention_params.value = attn_metadata.shared_v_cache;

      //   shared_attention_params.plan_info = attn_metadata.plan_info;
      //   // shared_attention_params.is_decode_shared = true;

      //   xllm::kernel::batch_prefill(shared_attention_params);
      // }
      
      // {
      //   LLM_NVTX_RANGE_COLOR("reshape_shared_output", 0xFFFF00FF);  // Magenta
      //   // batch_prefill的输出是 [batch_size, num_kv_heads_ * beam_size * group_size, head_size_]
      //   // 需要reshape回 [batch_size, num_kv_heads_, beam_size, group_size, head_size_]
      //   shared_o = shared_o.view({batch_size, num_kv_heads_, beam_size, group_size, head_size_});
      //   // permute回 [batch_size, beam_size, num_kv_heads_, group_size, head_size_]
      //   shared_o = shared_o.permute({0, 2, 1, 3, 4}).contiguous();
      //   // 然后reshape回原始的 [batch_size * beam_size, num_heads_, head_size_]
      //   shared_o = shared_o.view({batch_size * beam_size, num_heads_, head_size_});
        
      //   // batch_prefill的输出是 [batch_size, num_kv_heads_ * beam_size * group_size, 1]
      //   // 需要reshape回 [batch_size, num_kv_heads_, beam_size, group_size, 1]
      //   shared_lse = shared_lse.view({batch_size, num_kv_heads_, beam_size, group_size, 1});
      //   // permute回 [batch_size, beam_size, num_kv_heads_, group_size, 1]
      //   shared_lse = shared_lse.permute({0, 2, 1, 3, 4}).contiguous();
      //   // 然后reshape回原始的 [batch_size * beam_size, num_heads_, 1]
      //   shared_lse = shared_lse.view({batch_size * beam_size, num_heads_, 1});
      // }
      // // unshared

      // {
      //   LLM_NVTX_RANGE_COLOR("decoder_reshape_and_cache", 0xFF008080);  // Teal
      //   key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
      //   value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});
        
      //   xllm::kernel::cuda::decoder_reshape_and_cache(key, 
      //                                                 value, 
      //                                                 k_cache, 
      //                                                 v_cache, 
      //                                                 attn_metadata.block_table, 
      //                                                 attn_metadata.step);
      // }
      
      // torch::Tensor unshared_lse = attn_metadata.unshared_lse;
      // torch::Tensor unshared_o = attn_metadata.unshared_o;
      
      // {
      //   LLM_NVTX_RANGE_COLOR("batch_decode_unshared", 0xFFFF0000);  // Red
      //   xllm::kernel::AttentionParams unshared_attention_params;
      //   // auto unshared_lse = std::nullopt;
        
      //   unshared_attention_params.return_lse = true;
      //   unshared_attention_params.output_lse = unshared_lse;
        
      //   unshared_attention_params.window_size_left = sliding_window_;
      //   unshared_attention_params.scale = scale_;
      //   unshared_attention_params.compute_dtype = attn_metadata.compute_dtype;
      //   // for flashinfer
      //   unshared_attention_params.float_workspace_buffer =
      //       FlashinferWorkspace::get_instance().get_float_workspace_buffer();
      //   unshared_attention_params.int_workspace_buffer =
      //       FlashinferWorkspace::get_instance().get_int_workspace_buffer();
      //   unshared_attention_params.page_locked_int_workspace_buffer =
      //       FlashinferWorkspace::get_instance()
      //           .get_page_locked_int_workspace_buffer();

      //   // TODO: support chunked prefill
      //   CHECK(!attn_metadata.is_chunked_prefill)
      //       << "chunked prefill is not supported";
        
      //   // total_beams = batch_size * beam_size
      //   query = query.view({-1, 1, num_heads_, head_size_});
      //   unshared_o = unshared_o.view({-1, 1, num_heads_, head_size_});
      //   // LOG(INFO) << "query.shape: " << query.sizes();
      //   unshared_attention_params.query = query;
      //   unshared_attention_params.output = unshared_o;

      //   int64_t max_decode_step = k_cache.size(2);

      //   k_cache = k_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});
      //   v_cache = v_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});

      //   unshared_attention_params.k_cache = k_cache;
      //   unshared_attention_params.v_cache = v_cache;

      //   unshared_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      //   unshared_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
      //   unshared_attention_params.paged_kv_last_page_len = attn_metadata.paged_kv_last_page_len;

      //   xllm::kernel::batch_decode(unshared_attention_params);
      // }
      // // LOG(INFO) << "after kernel::batch_decode.";
      // // combine
      // {
      //   LLM_NVTX_RANGE_COLOR("lse_combine", 0xFF0000FF);  // Blue
      //   unshared_o = unshared_o.view({-1, num_heads_, head_size_});
      //   // LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      //   xllm::kernel::cuda::lse_combine(output, shared_o, shared_lse, unshared_o, unshared_lse);
      // }
      // LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
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