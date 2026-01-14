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

#include <glog/logging.h>

#include <unordered_map>

#include "core/common/global_flags.h"
#include "cuda_ops_api.h"
#include "function_factory.h"
#include "utils.h"

namespace xllm::kernel::cuda {

void batch_decode(const std::string& uri,
                  torch::Tensor plan_info,
                  torch::Tensor float_workspace_buffer,
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
                  bool use_tensor_core,
                  torch::Tensor kv_seq_lens,
                  std::optional<torch::Tensor> qo_indptr) {
  // Log plan_info device before use
  // if (plan_info.defined()) {
  //   LOG(INFO) << "batch_decode: plan_info device before kernel call: "
  //             << plan_info.device() << ", shape=" << plan_info.sizes()
  //             << ", dtype=" << plan_info.scalar_type()
  //             << ", uri=" << uri;
  // }
  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;
  }
  if (use_tensor_core) {
    torch::Tensor qo_indptr_to_use;
    if (qo_indptr.has_value()) {
      // Use provided qo_indptr from attn_metadata
      // TODO: consturct qo_indptr in CUDA graph execution
      qo_indptr_to_use = qo_indptr.value();
      VLOG(50) << "use provided qo_indptr in CUDA graph execution";
    } else {
      // Create qo_indptr if not provided (backward compatibility)
      const int64_t batch_size = paged_kv_last_page_len.size(0);
      torch::Tensor qo_indptr_host =
          get_cache_buffer(batch_size + 1, torch::kCPU);
      qo_indptr_to_use = qo_indptr_host.to(torch::kCUDA);
    }

    FunctionFactory::get_instance().fa2_prefill_paged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        k_cache,
        v_cache,
        qo_indptr_to_use,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        output,
        output_lse,
        /*mask_mode_code=*/0,  // NON_CAUSAL
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
  } else {
    VLOG(50) << "float_workspace_buffer.has_storage: "
             << float_workspace_buffer.has_storage();
    VLOG(50) << "int_workspace_buffer.has_storage: "
             << int_workspace_buffer.has_storage();
    VLOG(50) << "plan_info.has_storage: " << plan_info.has_storage();
    VLOG(50) << "query.has_storage: " << query.has_storage();
    VLOG(50) << "k_cache.has_storage: " << k_cache.has_storage();
    VLOG(50) << "v_cache.has_storage: " << v_cache.has_storage();
    VLOG(50) << "paged_kv_indptr.has_storage: "
             << paged_kv_indptr.has_storage();
    VLOG(50) << "paged_kv_indices.has_storage: "
             << paged_kv_indices.has_storage();
    VLOG(50) << "paged_kv_last_page_len.has_storage: "
             << paged_kv_last_page_len.has_storage();
    VLOG(50) << "output.has_storage: " << output.has_storage();
    FunctionFactory::get_instance().decode_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
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
