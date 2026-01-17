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

#include "xattention.h"

#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);

namespace xllm {
namespace layer {

std::tuple<torch::Tensor, std::optional<torch::Tensor>> XAttentionImpl::forward(
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

  if (attn_metadata.is_prefill) {
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";

    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    bool causal = attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
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
        /*block_size*/ k_cache.size(1),
        /*window_size_left*/ sliding_window_,
        /*enable_cuda_graph*/ false,
        /*causal*/ causal,
        /*use_tensor_core*/ true);

    xllm::kernel::cuda::prefill_reshape_and_cache(
        key, value, attn_metadata.full_k_cache, attn_metadata.full_v_cache);
    xllm::kernel::AttentionParams attention_params(attn_metadata);
    attention_params.query = query;
    attention_params.output = output_tensor;
    attention_params.output_lse = output_lse;
    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    // for flashinfer
    attention_params.float_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();
    // attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
    // attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;

    attention_params.key = key;
    attention_params.value = value;
    // attention_params.uri = attn_metadata.plan_info->uri;
    // attention_params.plan_info = attn_metadata.plan_info->plan_info;
    xllm::kernel::batch_prefill(attention_params);
  } else {
    // uint32_t batch_size = attn_metadata.paged_kv_last_page_len.numel();
    uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
    uint32_t total_beam = query.size(0);
    uint32_t beam_size = total_beam / batch_size;

    // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
    key = key.view({batch_size, beam_size, num_kv_heads_, head_size_})
              .contiguous();
    value = value.view({batch_size, beam_size, num_kv_heads_, head_size_})
                .contiguous();
    xllm::kernel::cuda::decoder_reshape_and_cache(
        key,
        value,
        attn_metadata.unshared_k_cache,
        attn_metadata.unshared_v_cache,
        attn_metadata.naive_block_table,
        attn_metadata.step);

    torch::Tensor full_k_cache = attn_metadata.full_k_cache.unsqueeze(1);
    torch::Tensor full_v_cache = attn_metadata.full_v_cache.unsqueeze(1);

    // maybe we need to update shared attn state before execute attention,
    // currently we update flashinfer step_wise_attn_state_ at layer 0.
    if (attn_metadata.enable_cuda_graph) {
      CHECK(attn_metadata.plan_info->plan_info.defined())
          << "plan_info plan_info should not be null when enable_cuda_graph is "
             "true";
      VLOG(50) << "no need to update plan_info for CUDA graph";
    } else {
      std::string backend = "fa3";
      flashinfer::update_plan_info(attn_metadata.plan_info,
                                   backend,
                                   attn_metadata,
                                   query.scalar_type(),
                                   key.scalar_type(),
                                   output_tensor.scalar_type(),
                                   head_size_,
                                   head_size_,
                                   num_heads_,
                                   num_kv_heads_,
                                   /*block_size*/ full_k_cache.size(1),
                                   /*window_size_left*/ sliding_window_,
                                   /*enable_cuda_graph*/ false,
                                   /*causal*/ false,
                                   /*use_tensor_core*/ false);
    }

    xllm::kernel::AttentionParams attention_params(attn_metadata);
    auto unshared_lse = std::nullopt;

    attention_params.return_lse = false;
    attention_params.output_lse = unshared_lse;

    attention_params.window_size_left = sliding_window_;
    attention_params.scale = scale_;
    // attention_params.compute_dtype = attn_metadata.compute_dtype;
    // for flashinfer
    attention_params.float_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_float_workspace_buffer();
    attention_params.int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_int_workspace_buffer();
    attention_params.page_locked_int_workspace_buffer =
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .get_page_locked_int_workspace_buffer();
    // TODO: support chunked prefill
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";

    attention_params.query = query;
    attention_params.output = output_tensor;

    attention_params.k_cache = full_k_cache;
    attention_params.v_cache = full_v_cache;

    // attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
    // attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
    // attention_params.paged_kv_last_page_len =
    //     attn_metadata.paged_kv_last_page_len;
    // attention_params.uri = attn_metadata.plan_info->uri;
    // attention_params.plan_info = attn_metadata.plan_info->plan_info;
    // attention_params.use_tensor_core = false;
    const_cast<AttentionMetadata&>(attention_params.attn_metadata)
        .use_tensor_core = false;
    xllm::kernel::batch_decode(attention_params);
  }
  output_tensor = output_tensor.view({-1, num_heads_ * head_size_});
  return {output_tensor, output_lse};
}

}  // namespace layer
}  // namespace xllm