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

#include "cuda_ops_api.h"
#include "common/nvtx_helper.h"
#include "function_factory.h"

namespace xllm::kernel::cuda {

void batch_prefill(torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph,
                   std::optional<torch::Tensor>& plan_info,
                   bool is_decode_shared) {
  LLM_NVTX_RANGE("batch_prefill");
  
  std::string backend;
  std::string uri;
  {
    LLM_NVTX_RANGE_COLOR("batch_prefill_prepare", 0xFF808080);  // Gray
    backend = determine_attention_backend(/*pos_encoding_mode=*/0,
                                          /*use_fp16_qk_reduction=*/false,
                                          /*use_custom_mask=*/false);

    uri = get_batch_prefill_uri(backend,
                                query.scalar_type(),
                                key.scalar_type(),
                                output.scalar_type(),
                                q_cu_seq_lens.scalar_type(),
                                query.size(-1),
                                value.size(-1),
                                /*pos_encoding_mode=*/0,
                                /*use_sliding_window=*/false,
                                /*use_logits_soft_cap=*/false,
                                /*use_fp16_qk_reduction=*/false);
  }

  torch::Tensor qo_indptr_host;
  torch::Tensor kv_cu_seq_lens_host;
  torch::Tensor kv_len_arr_host;
  int64_t total_num_rows;
  int64_t batch_size;
  {
    LLM_NVTX_RANGE_COLOR("batch_prefill_d2h_memcpy", 0xFFFF00FF);  // Magenta
    qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
    kv_cu_seq_lens_host = kv_cu_seq_lens.to(torch::kCPU);
  }
  
  {
    LLM_NVTX_RANGE_COLOR("batch_prefill_cpu_compute", 0xFF800080);  // Purple
    kv_len_arr_host =
        kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
    total_num_rows = qo_indptr_host[-1].item<int64_t>();
    batch_size = qo_indptr_host.size(0) - 1;
  }

  // 使用传入的 plan_info（存储在 AttentionMetadata 中），每个 batch/stream 有独立的 plan_info
  // 如果没有传入，则在这里生成（fallback）
  torch::Tensor plan_info_tensor;
  if (plan_info.has_value()) {
    // LOG(INFO) << "plan_info is already precomputed";
    // plan_info 已经预先计算好，直接复用
    plan_info_tensor = *plan_info;
    // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
  } else {
    // LOG(INFO) << "plan_info is not precomputed, generating it";
    // plan_info 未预先计算，需要在这里计算（fallback）
    LLM_NVTX_RANGE_COLOR("batch_prefill_plan", 0xFF00FF00);  // Green
    plan_info_tensor = FunctionFactory::get_instance().prefill_plan_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        page_locked_int_workspace_buffer,
        qo_indptr_host,
        kv_cu_seq_lens_host,
        kv_len_arr_host,
        total_num_rows,
        batch_size,
        query.size(1),  // num_qo_heads
        key.size(1),    // num_kv_heads
        /*page_size=*/128,
        enable_cuda_graph,
        query.size(-1),  // head_dim_qk
        value.size(-1),  // head_dim_vo
        /*causal=*/true);
    // Only print debug logs for decode shared calls
    // LOG(INFO) << "float_workspace_buffer.shape: " << float_workspace_buffer.sizes();
    // LOG(INFO) << "int_workspace_buffer.shape: " << int_workspace_buffer.sizes();
    // LOG(INFO) << "page_locked_int_workspace_buffer.shape: " << page_locked_int_workspace_buffer.sizes();
    // LOG(INFO) << "qo_indptr_host: " << qo_indptr_host;
    // LOG(INFO) << "kv_cu_seq_lens_host: " << kv_cu_seq_lens_host;
    // LOG(INFO) << "kv_len_arr_host: " << kv_len_arr_host;
    // LOG(INFO) << "total_num_rows: " << total_num_rows;
    // LOG(INFO) << "batch_size: " << batch_size;
    // LOG(INFO) << "num_qo_heads: " << query.size(1);
    // LOG(INFO) << "num_kv_heads: " << key.size(1);
    // LOG(INFO) << "page_size: " << /*page_size=*/128;
    // LOG(INFO) << "enable_cuda_graph: " << enable_cuda_graph;
    // LOG(INFO) << "head_dim_qk: " << query.size(-1);
    // LOG(INFO) << "head_dim_vo: " << value.size(-1);
    // LOG(INFO) << "causal: " << /*causal=*/true;
    // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
    // LOG(FATAL) << "after batch_prefill_plan";
    
  }

  {
    LLM_NVTX_RANGE_COLOR("batch_prefill_kernel", 0xFFFF0000);  // Red
    if (backend == "fa2") {
      FunctionFactory::get_instance().fa2_prefill_ragged_run_func(uri).call(
          float_workspace_buffer,
          int_workspace_buffer,
          plan_info_tensor,
          query,
          key,
          value,
          q_cu_seq_lens,
          kv_cu_seq_lens,
          output,
          output_lse,
          /*mask_mode_code=*/1,  // CAUSAL
          /*kv_layout_code=*/0,  // NHD layout
          window_left,
          support_pdl(),
          /*maybe_custom_mask=*/std::optional<torch::Tensor>(),
          /*maybe_mask_indptr=*/std::optional<torch::Tensor>(),
          /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
          /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
          /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
          /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
          /*logits_soft_cap=*/0.0,
          sm_scale,
          /*rope_rcp_scale=*/1.0,
          /*rope_rcp_theta=*/1.0 / 10000.0,
          /*token_pos_in_items_len=*/0);
    } else if (backend == "fa3") {
      FunctionFactory::get_instance().fa3_prefill_ragged_run_func(uri).call(
          float_workspace_buffer,
          int_workspace_buffer,
          plan_info_tensor,
          query,
          key,
          value,
          q_cu_seq_lens,
          kv_cu_seq_lens,
          output,
          output_lse,
          /*mask_mode_code=*/1,  // CAUSAL
          /*kv_layout_code=*/0,  // NHD layout
          window_left,
          support_pdl(),
          /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
          /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
          /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
          /*logits_soft_cap=*/0.0,
          sm_scale,
          /*token_pos_in_items_len=*/0);
    }
  }
}

torch::Tensor generate_prefill_plan_info(
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    torch::ScalarType dtype_q,
    torch::ScalarType dtype_kv,
    torch::ScalarType dtype_o,
    bool enable_cuda_graph) {
  LLM_NVTX_RANGE("generate_prefill_plan_info");
  
  
  
  std::string backend;
  std::string uri;
  {
    LLM_NVTX_RANGE_COLOR("generate_plan_prepare", 0xFF808080);  // Gray
    backend = determine_attention_backend(/*pos_encoding_mode=*/0,
                                          /*use_fp16_qk_reduction=*/false,
                                          /*use_custom_mask=*/false);

    uri = get_batch_prefill_uri(backend,
                                dtype_q,
                                dtype_kv,
                                dtype_o,
                                q_cu_seq_lens.scalar_type(),
                                head_dim_qk,
                                head_dim_vo,
                                /*pos_encoding_mode=*/0,
                                /*use_sliding_window=*/false,
                                /*use_logits_soft_cap=*/false,
                                /*use_fp16_qk_reduction=*/false);
  }

  torch::Tensor qo_indptr_host;
  torch::Tensor kv_cu_seq_lens_host;
  torch::Tensor kv_len_arr_host;
  int64_t total_num_rows;
  int64_t batch_size;
  {
    LLM_NVTX_RANGE_COLOR("generate_plan_d2h_memcpy", 0xFFFF00FF);  // Magenta
    qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
    kv_cu_seq_lens_host = kv_cu_seq_lens.to(torch::kCPU);
  }
  
  {
    LLM_NVTX_RANGE_COLOR("generate_plan_cpu_compute", 0xFF800080);  // Purple
    kv_len_arr_host =
        kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
    total_num_rows = qo_indptr_host[-1].item<int64_t>();
    batch_size = qo_indptr_host.size(0) - 1;
  }
  
  {
    LLM_NVTX_RANGE_COLOR("generate_plan_call", 0xFF00FF00);  // Green
    torch::Tensor plan_info_tensor = FunctionFactory::get_instance().prefill_plan_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        page_locked_int_workspace_buffer,
        qo_indptr_host,
        kv_cu_seq_lens_host,
        kv_len_arr_host,
        total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        /*page_size=*/128,
        enable_cuda_graph,
        head_dim_qk,
        head_dim_vo,
        /*causal=*/true);
    // LOG(INFO) << "float_workspace_buffer.shape: " << float_workspace_buffer.sizes();
    // LOG(INFO) << "int_workspace_buffer.shape: " << int_workspace_buffer.sizes();
    // LOG(INFO) << "page_locked_int_workspace_buffer.shape: " << page_locked_int_workspace_buffer.sizes();
    // LOG(INFO) << "qo_indptr_host: " << qo_indptr_host;
    // LOG(INFO) << "kv_cu_seq_lens_host: " << kv_cu_seq_lens_host;
    // LOG(INFO) << "kv_len_arr_host: " << kv_len_arr_host;
    // LOG(INFO) << "total_num_rows: " << total_num_rows;
    // LOG(INFO) << "batch_size: " << batch_size;
    // LOG(INFO) << "num_qo_heads: " << num_qo_heads;
    // LOG(INFO) << "num_kv_heads: " << num_kv_heads;
    // LOG(INFO) << "page_size: " << /*page_size=*/128;
    // LOG(INFO) << "enable_cuda_graph: " << enable_cuda_graph;
    // LOG(INFO) << "head_dim_qk: " << head_dim_qk;
    // LOG(INFO) << "head_dim_vo: " << head_dim_vo;
    // LOG(INFO) << "causal: " << /*causal=*/true;
    // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
    // LOG(FATAL) << "after generate_plan_call";
    return plan_info_tensor;
  }
}

}  // namespace xllm::kernel::cuda
