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

#include "npu_reshape_and_cache_impl.h"

#include <glog/logging.h>

namespace xllm::kernel {

void NpuReshapeAndCacheImpl::param_from_args(
    atb::infer::ReshapeAndCacheParam& param,
    const ModelArgs& args) {
  // Use default ReshapeAndCacheParam settings
}

int64_t NpuReshapeAndCacheImpl::init_node(
    atb_speed::Model::Node& node,
    atb::infer::ReshapeAndCacheParam& param) {
  name_ = "reshape_and_cache";
  model_name_ = "llm";
  run_task_func_ = std::bind(&NpuReshapeAndCacheImpl::run_task,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2);

  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb::CreateOperation(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }

  return atb::NO_ERROR;
}

NpuReshapeAndCacheImpl::NpuReshapeAndCacheImpl(const ModelContext& context)
    : NpuBaseLayer(context) {
  param_from_args(reshape_and_cache_param_, context.get_model_args());

  atb::Status status =
      init_node(reshape_and_cache_node_, reshape_and_cache_param_);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to initialize node, status: " << status;
    throw std::runtime_error(
        "NpuReshapeAndCacheImpl initialization failed with status: " +
        std::to_string(status));
  }
}

void NpuReshapeAndCacheImpl::verify_loaded_weights(
    const std::string weight_str) const {
  // No operations are needed for this layer
}

void NpuReshapeAndCacheImpl::merge_loaded_weights() {
  // No operations are needed for this layer
}

void NpuReshapeAndCacheImpl::load_state_dict(const StateDict& state_dict) {
  // No operations are needed for this layer
}

void NpuReshapeAndCacheImpl::forward(torch::Tensor& intermediate_k,
                                     torch::Tensor& intermediate_v,
                                     torch::Tensor& in_k_cache,
                                     torch::Tensor& in_v_cache,
                                     torch::Tensor& in_slots,
                                     int nodeId) {
  atb::Status st;
  build_node_variant_pack(reshape_and_cache_node_,
                          intermediate_k,
                          intermediate_v,
                          in_k_cache,
                          in_v_cache,
                          in_slots);
  st = execute_node(reshape_and_cache_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
}

void NpuReshapeAndCacheImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
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
