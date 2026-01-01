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

#pragma once

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <glog/logging.h>

#include <optional>

#include "utils.h"

namespace xllm::kernel::cuda {

// TODO: add head_size parameter
void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      // int64_t head_size,
                      bool is_neox);

// act_mode only support silu, gelu, gelu_tanh
void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache);

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
                  bool is_decode_shared = false);

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
                  std::optional<torch::Tensor>& plan_info);

void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

void lse_combine(torch::Tensor output,
                 torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse);

void decoder_reshape_and_cache(torch::Tensor proj_k,
                                torch::Tensor proj_v,
                                torch::Tensor unshared_k_cache,
                                torch::Tensor unshared_v_cache,
                                torch::Tensor block_table,
                                uint32_t step);

// Generate plan_info for batch_prefill optimization
// This should be called once before the layer loop for prefill mode
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
    bool enable_cuda_graph);

// Generate plan_info for batch_decode optimization
// This should be called once before the layer loop for decode mode
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
    bool enable_cuda_graph);

}  // namespace xllm::kernel::cuda
