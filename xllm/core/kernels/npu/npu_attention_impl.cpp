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

#include "npu_attention_impl.h"

#include <glog/logging.h>

namespace xllm::kernel {

int64_t NpuAttentionImpl::init_node(atb_speed::Model::Node& node,
                                    const std::string& opType,
                                    const std::string& opParam,
                                    const std::string& opName) {
  model_name_ = opName;
  run_task_func_ = std::bind(&NpuAttentionImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);

  auto baseOperation = atb_torch::BaseOperation(opType, opParam, opName);
  node.operation.reset(baseOperation.GetAtbOperation());
  return atb::NO_ERROR;
}

NpuAttentionImpl::NpuAttentionImpl(const std::string& opType,
                                   const std::string& opParam,
                                   const std::string& opName) {
  atb::Status status = init_node(attention_node_, opType, opParam, opName);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error(
        "NpuAttentionImpl initialization failed with status: " +
        std::to_string(status));
  }
}

void NpuAttentionImpl::verify_loaded_weights(
    const std::string weight_str) const {
  // No operations are needed for this layer
}

void NpuAttentionImpl::merge_loaded_weights() {
  // No operations are needed for this layer
}

void NpuAttentionImpl::load_state_dict(const StateDict& state_dict) {
  // No operations are needed for this layer
}

void NpuAttentionImpl::forward(torch::Tensor& intermediate_k,
                               torch::Tensor& intermediate_v,
                               torch::Tensor& in_k_cache,
                               torch::Tensor& in_v_cache,
                               torch::Tensor& in_slots,
                               int nodeId) {
  atb::Status st;
  build_node_variant_pack(attention_node_,
                          intermediate_k,
                          intermediate_v,
                          in_k_cache,
                          in_v_cache,
                          in_slots);
  st = execute_node(attention_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
}

void NpuAttentionImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                               torch::Tensor& intermediate_k,
                                               torch::Tensor& intermediate_v,
                                               torch::Tensor& in_k_cache,
                                               torch::Tensor& in_v_cache,
                                               torch::Tensor& in_slots) {
  internal_intermediate_k = atb_speed::Utils::AtTensor2Tensor(intermediate_k);
  internal_intermediate_v = atb_speed::Utils::AtTensor2Tensor(intermediate_v);
  internal_in_k_cache = atb_speed::Utils::AtTensor2Tensor(in_k_cache);
  internal_in_v_cache = atb_speed::Utils::AtTensor2Tensor(in_v_cache);
  internal_in_slots = atb_speed::Utils::AtTensor2Tensor(in_slots);

  atb::SVector<atb::Tensor> ins = {internal_intermediate_k,
                                   internal_intermediate_v,
                                   internal_in_k_cache,
                                   internal_in_v_cache,
                                   internal_in_slots};
  atb::SVector<atb::Tensor> outs = {internal_in_k_cache,
                                    internal_in_v_cache};  // write in place

  node.variantPack.inTensors = ins;
  node.variantPack.outTensors = outs;
}

}  // namespace xllm::kernel
