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

#include "dense_mlp.h"

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "kernels/mlu/torch_ops_api.h"

namespace xllm {
namespace layer {

DenseMLPImpl::DenseMLPImpl(int hidden_size,
                           int intermediate_size,
                           bool is_gated,
                           bool has_bias,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : is_gated_(is_gated),
      intermediate_size_(intermediate_size),
      parallel_args_(parallel_args) {
  // 1. gate + up
  gate_up_proj_ =
      register_module("gate_up_proj",
                      ColumnParallelLinearV1(hidden_size,
                                             intermediate_size_ * 2,
                                             /*bias=*/has_bias,
                                             /*gather_output=*/false,
                                             parallel_args,
                                             options));

  // 2. down
  down_proj_ = register_module("down_proj",
                               RowParallelLinear(intermediate_size_,
                                                 hidden_size,
                                                 /*bias=*/has_bias,
                                                 /*input_is_parallelized=*/true,
                                                 /*if_reduce_results=*/true,
                                                 parallel_args,
                                                 options));
}

torch::Tensor DenseMLPImpl::forward(const torch::Tensor& hidden_states) {
  // input shape: [num_tokens, hidden_size]
  auto gate_up = gate_up_proj_->forward(hidden_states);

  int64_t batch_size = gate_up.sizes()[0];
  // auto output = torch::empty(
  //     {batch_size, intermediate_size_ / parallel_args_.world_size()},
  //     gate_up.options());

  auto output1 = at_npu::native::custom_ops::npu_swiglu(gate_up);
  // tmo::torch_api::active(gate_up,
  //                        output,
  //                        std::nullopt /* bias */,
  //                        std::nullopt /* cusum_token_count */,
  //                        xllm::mlu::kActModeSilu,
  //                        is_gated_,
  //                        0 /* start_expert_id */,
  //                        0 /* expert_size */);

  return down_proj_->forward(output1);
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict) {
  gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
}

}  // namespace layer
}  // namespace xllm