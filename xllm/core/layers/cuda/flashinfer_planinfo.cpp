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

#include "flashinfer_planinfo.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/util/utils.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"
namespace xllm::layer::flashinfer {

void update_plan_info(std::shared_ptr<PlanInfo> plan_info,
                      const std::string& backend,
                      const AttentionMetadata& attn_meta,
                      c10::ScalarType query_dtype,
                      c10::ScalarType key_dtype,
                      c10::ScalarType output_dtype,
                      int32_t head_dim_qk,
                      int32_t head_dim_vo,
                      int32_t num_qo_heads,
                      int32_t num_kv_heads,
                      int32_t block_size,
                      int32_t window_size_left,
                      bool enable_cuda_graph,
                      bool causal,
                      bool use_tensor_core) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->layer_id != 0) return;

  VLOG(kGraphExecutorLogVerboseLevel)
      << "update_plan_info: layer_id=" << plan_info->layer_id
      << ", enable_cuda_graph=" << enable_cuda_graph;
  // 1. prefill plan info
  if (causal) {
    plan_info->uri = kernel::cuda::get_batch_prefill_uri(
        backend,
        query_dtype,
        key_dtype,
        output_dtype,
        attn_meta.q_cu_seq_lens.scalar_type(),
        head_dim_qk,
        head_dim_vo,
        /*pos_encoding_mode=*/0,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_fp16_qk_reduction=*/false);

    torch::Tensor qo_indptr_host = attn_meta.q_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_cu_seq_lens_host =
        attn_meta.kv_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_len_arr_host =
        kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
    const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
    const int64_t batch_size = qo_indptr_host.size(0) - 1;
    auto call_plan_func = [&](auto&& func) {
      return func.call(
          FlashinferWorkspace::get_instance().get_float_workspace_buffer(),
          FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
          FlashinferWorkspace::get_instance()
              .get_page_locked_int_workspace_buffer(),
          qo_indptr_host,
          kv_cu_seq_lens_host,
          kv_len_arr_host,
          total_num_rows,
          batch_size,
          num_qo_heads,
          num_kv_heads,
          /*page_size=*/1,
          enable_cuda_graph,
          head_dim_qk,
          head_dim_vo,
          causal);
    };
    if (backend == "fa2") {
      plan_info->plan_info = call_plan_func(
          kernel::cuda::FunctionFactory::get_instance().fa2_prefill_plan_func(
              plan_info->uri));
    } else {
      plan_info->plan_info = call_plan_func(
          kernel::cuda::FunctionFactory::get_instance().fa3_prefill_plan_func(
              plan_info->uri));
    }
  } else {
    // 2. decode plan info
    if (use_tensor_core) {
      plan_info->uri = kernel::cuda::get_batch_prefill_uri(
          /*backend=*/"fa2",
          query_dtype,
          key_dtype,
          output_dtype,
          attn_meta.paged_kv_indptr.scalar_type(),
          head_dim_qk,
          head_dim_vo,
          /*pos_encoding_mode=*/0,
          /*use_sliding_window=*/false,
          /*use_logits_soft_cap=*/false,
          /*use_fp16_qk_reduction=*/false);
      const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
      torch::Tensor qo_indptr_host =
          kernel::cuda::get_cache_buffer(batch_size + 1, torch::kCPU);
      torch::Tensor qo_indptr = qo_indptr_host.to(torch::kCUDA);
      torch::Tensor paged_kv_indptr_host =
          attn_meta.paged_kv_indptr.to(torch::kCPU);
      torch::Tensor kv_len_arr_host = attn_meta.kv_seq_lens.to(torch::kCPU);
      if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
        VLOG(kGraphExecutorLogVerboseLevel)
            << "use_tensor_core: " << use_tensor_core;
        VLOG(kGraphExecutorLogVerboseLevel) << "batch_size: " << batch_size;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "qo_indptr_host: " << qo_indptr_host;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "paged_kv_indptr_host: " << paged_kv_indptr_host;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "kv_len_arr_host: " << kv_len_arr_host;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "enable_cuda_graph: " << enable_cuda_graph;
        VLOG(kGraphExecutorLogVerboseLevel) << "head_dim_qk: " << head_dim_qk;
        VLOG(kGraphExecutorLogVerboseLevel) << "head_dim_vo: " << head_dim_vo;
        VLOG(kGraphExecutorLogVerboseLevel) << "num_qo_heads: " << num_qo_heads;
        VLOG(kGraphExecutorLogVerboseLevel) << "num_kv_heads: " << num_kv_heads;
        VLOG(kGraphExecutorLogVerboseLevel) << "block_size: " << block_size;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "window_size_left: " << window_size_left;
        VLOG(kGraphExecutorLogVerboseLevel) << "query_dtype: " << query_dtype;
        VLOG(kGraphExecutorLogVerboseLevel) << "key_dtype: " << key_dtype;
      }
      plan_info->plan_info =
          kernel::cuda::FunctionFactory::get_instance()
              .fa2_prefill_plan_func(plan_info->uri)
              .call(FlashinferWorkspace::get_instance()
                        .get_float_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_int_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer(),
                    qo_indptr_host,
                    paged_kv_indptr_host,
                    kv_len_arr_host,
                    batch_size,  // total_num_rows
                    batch_size,
                    num_qo_heads,  // num_qo_heads
                    num_kv_heads,  // num_kv_heads
                    block_size,    // block_size
                    enable_cuda_graph,
                    head_dim_qk,  // head_dim_qk
                    head_dim_vo,  // head_dim_vo
                    /*causal=*/false);
    } else {
      plan_info->uri = kernel::cuda::get_batch_decode_uri(
          query_dtype,
          key_dtype,
          output_dtype,
          attn_meta.paged_kv_indptr.scalar_type(),
          head_dim_qk,
          head_dim_vo,
          /*pos_encoding_mode=*/0,
          /*use_sliding_window=*/false,
          /*use_logits_soft_cap=*/false);
      torch::Tensor paged_kv_indptr_host =
          attn_meta.paged_kv_indptr.to(torch::kCPU);
      const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
      torch::Tensor empty_q_data =
          torch::empty({0}, torch::TensorOptions().dtype(query_dtype));
      torch::Tensor empty_kv_data =
          torch::empty({0}, torch::TensorOptions().dtype(key_dtype));
      plan_info->plan_info =
          kernel::cuda::FunctionFactory::get_instance()
              .decode_plan_func(plan_info->uri)
              .call(FlashinferWorkspace::get_instance()
                        .get_float_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_int_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer(),
                    paged_kv_indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    block_size,
                    enable_cuda_graph,
                    window_size_left,
                    /*logits_soft_cap=*/0.0,
                    head_dim_qk,
                    head_dim_vo,
                    empty_q_data,
                    empty_kv_data);
      if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
        VLOG(kGraphExecutorLogVerboseLevel)
            << "use_tensor_core: " << use_tensor_core;
        VLOG(kGraphExecutorLogVerboseLevel) << "batch_size: " << batch_size;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "paged_kv_indptr_host: " << paged_kv_indptr_host;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "enable_cuda_graph: " << enable_cuda_graph;
        VLOG(kGraphExecutorLogVerboseLevel) << "head_dim_qk: " << head_dim_qk;
        VLOG(kGraphExecutorLogVerboseLevel) << "head_dim_vo: " << head_dim_vo;
        VLOG(kGraphExecutorLogVerboseLevel) << "num_qo_heads: " << num_qo_heads;
        VLOG(kGraphExecutorLogVerboseLevel) << "num_kv_heads: " << num_kv_heads;
        VLOG(kGraphExecutorLogVerboseLevel) << "block_size: " << block_size;
        VLOG(kGraphExecutorLogVerboseLevel)
            << "window_size_left: " << window_size_left;
        VLOG(kGraphExecutorLogVerboseLevel) << "query_dtype: " << query_dtype;
        VLOG(kGraphExecutorLogVerboseLevel) << "key_dtype: " << key_dtype;
      }
    }
  }

  // Log plan_info tensor information
  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    if (plan_info->plan_info.defined()) {
      std::string mode_str;
      if (causal) {
        mode_str = "prefill, " + backend;
      } else if (use_tensor_core) {
        mode_str = "decode, tensor_core";
      } else {
        mode_str = "decode, non_tensor_core";
      }
      LOG(INFO) << "plan_info (" << mode_str << "): "
                << "shape=" << plan_info->plan_info.sizes()
                << ", dtype=" << plan_info->plan_info.scalar_type()
                << ", device=" << plan_info->plan_info.device()
                << ", numel=" << plan_info->plan_info.numel() << ", num_bytes="
                << plan_info->plan_info.numel() *
                       plan_info->plan_info.element_size()
                << ", uri=" << plan_info->uri;
      VLOG(kGraphExecutorLogVerboseLevel)
          << "plan_info: " << plan_info->plan_info;
    }
  }
}

}  // namespace xllm::layer::flashinfer
