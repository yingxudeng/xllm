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

torch::Tensor generate_decode_plan_info(
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_last_page_len,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t window_left,
    bool enable_cuda_graph) {
  LLM_NVTX_RANGE("generate_decode_plan_info");
  // LOG(INFO) << "in generate_decode_plan_info";
  std::string uri;
  {
    LLM_NVTX_RANGE_COLOR("generate_decode_plan_prepare", 0xFF808080);  // Gray
    uri = get_batch_decode_uri(query.scalar_type(),
                               k_cache.scalar_type(),
                               query.scalar_type(),  // output dtype same as query
                               paged_kv_indptr.scalar_type(),
                               query.size(-1),
                               v_cache.size(-1),
                               /*pos_encoding_mode=*/0,
                               /*use_sliding_window=*/false,
                               /*use_logits_soft_cap=*/false);
  }

  torch::Tensor paged_kv_indptr_host;
  int64_t batch_size;
  {
    LLM_NVTX_RANGE_COLOR("generate_decode_plan_d2h_memcpy", 0xFFFF00FF);  // Magenta
    paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
    batch_size = paged_kv_last_page_len.size(0);
  }

  torch::Tensor empty_q_data;
  torch::Tensor empty_kv_data;
  {
    LLM_NVTX_RANGE_COLOR("generate_decode_plan_prepare_tensors", 0xFF800080);  // Purple
    empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query.scalar_type()));
    empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(k_cache.scalar_type()));
  }
  
  // LOG(INFO) << "float_workspace_buffer.shape: " << float_workspace_buffer.sizes();
  // LOG(INFO) << "int_workspace_buffer.shape: " << int_workspace_buffer.sizes();
  // LOG(INFO) << "page_locked_int_workspace_buffer.shape: " << page_locked_int_workspace_buffer.sizes();
  // LOG(INFO) << "paged_kv_indptr_host: " << paged_kv_indptr_host;
  // LOG(INFO) << "batch_size: " << batch_size;
  // LOG(INFO) << "num_qo_heads: " << query.size(1);
  // LOG(INFO) << "num_kv_heads: " << k_cache.size(2);
  // LOG(INFO) << "block_size: " << k_cache.size(1);
  // LOG(INFO) << "enable_cuda_graph: " << enable_cuda_graph;
  // LOG(INFO) << "window_left: " << window_left;
  // LOG(INFO) << "logits_soft_cap: " << 0.0;
  // LOG(INFO) << "head_dim_qk: " << query.size(-1);
  // LOG(INFO) << "head_dim_vo: " << v_cache.size(-1);
  // LOG(INFO) << "empty_q_data: " << empty_q_data;
  // LOG(INFO) << "empty_kv_data: " << empty_kv_data;
  {
    LLM_NVTX_RANGE_COLOR("generate_decode_plan_call", 0xFF00FF00);  // Green
    torch::Tensor plan_info_tensor = FunctionFactory::get_instance().decode_plan_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        page_locked_int_workspace_buffer,
        paged_kv_indptr_host,
        batch_size,
        query.size(1),    // num_qo_heads
        k_cache.size(2),  // num_kv_heads
        k_cache.size(1),  // block_size
        enable_cuda_graph,
        window_left,
        /*logits_soft_cap=*/0.0,
        query.size(-1),    // head_dim_qk
        v_cache.size(-1),  // head_dim_vo
        empty_q_data,
        empty_kv_data);
  // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
    return plan_info_tensor;
  }
}

void batch_decode(torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool enable_cuda_graph,
                  std::optional<torch::Tensor>& plan_info) {
  LLM_NVTX_RANGE("batch_decode");
  
  std::string uri;
  {
    LLM_NVTX_RANGE_COLOR("batch_decode_prepare", 0xFF808080);  // Gray
    uri = get_batch_decode_uri(query.scalar_type(),
                               k_cache.scalar_type(),
                               output.scalar_type(),
                               paged_kv_indptr.scalar_type(),
                               query.size(-1),
                               v_cache.size(-1),
                               /*pos_encoding_mode=*/0,
                               /*use_sliding_window=*/false,
                               /*use_logits_soft_cap=*/false);
  }

  torch::Tensor paged_kv_indptr_host;
  int64_t batch_size;
  {
    LLM_NVTX_RANGE_COLOR("batch_decode_d2h_memcpy", 0xFFFF00FF);  // Magenta
    paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
    batch_size = paged_kv_last_page_len.size(0);
  }

  torch::Tensor empty_q_data;
  torch::Tensor empty_kv_data;
  {
    LLM_NVTX_RANGE_COLOR("batch_decode_prepare_tensors", 0xFF800080);  // Purple
    empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query.scalar_type()));
    empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(k_cache.scalar_type()));
  }

  torch::Tensor plan_info_tensor;
  if (plan_info.has_value()) {
    // LOG(INFO) << "plan_info is already precomputed";
    // plan_info 已经预先计算好，直接复用
    plan_info_tensor = *plan_info;
    // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
  } else {
    // LOG(INFO) << "plan_info is not precomputed, generating it";
    // plan_info 未预先计算，需要在这里计算（fallback）
    // 使用原来的单例调用方式，不依赖新的 generate_decode_plan_info 函数
    LLM_NVTX_RANGE_COLOR("batch_decode_plan", 0xFF00FF00);  // Green
    plan_info_tensor = FunctionFactory::get_instance().decode_plan_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        page_locked_int_workspace_buffer,
        paged_kv_indptr_host,
        batch_size,
        query.size(1),    // num_qo_heads
        k_cache.size(2),  // num_kv_heads
        k_cache.size(1),  // block_size
        enable_cuda_graph,
        window_left,
        /*logits_soft_cap=*/0.0,
        query.size(-1),    // head_dim_qk
        v_cache.size(-1),  // head_dim_vo
        empty_q_data,
        empty_kv_data);
    // LOG(INFO) << "plan_info_tensor: " << plan_info_tensor;
    // LOG(INFO) << "float_workspace_buffer.shape: " << float_workspace_buffer.sizes();
    // LOG(INFO) << "int_workspace_buffer.shape: " << int_workspace_buffer.sizes();
    // LOG(INFO) << "page_locked_int_workspace_buffer.shape: " << page_locked_int_workspace_buffer.sizes();
    // LOG(INFO) << "paged_kv_indptr_host: " << paged_kv_indptr_host;
    // LOG(INFO) << "batch_size: " << batch_size;
    // LOG(INFO) << "num_qo_heads: " << query.size(1);
    // LOG(INFO) << "num_kv_heads: " << k_cache.size(2);
    // LOG(INFO) << "block_size: " << k_cache.size(1);
    // LOG(INFO) << "enable_cuda_graph: " << enable_cuda_graph;
    // LOG(INFO) << "window_left: " << window_left;
    // LOG(INFO) << "logits_soft_cap: " << 0.0;
    // LOG(INFO) << "head_dim_qk: " << query.size(-1);
    // LOG(INFO) << "head_dim_vo: " << v_cache.size(-1);
    // LOG(INFO) << "empty_q_data: " << empty_q_data;
    // LOG(INFO) << "empty_kv_data: " << empty_kv_data;
  }

  {
    LLM_NVTX_RANGE_COLOR("batch_decode_kernel", 0xFFFF0000);  // Red
    FunctionFactory::get_instance().decode_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info_tensor,
        query,
        k_cache,
        v_cache,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        output,
        output_lse,
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0);
  }
}

}  // namespace xllm::kernel::cuda
