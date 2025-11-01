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
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "layers/npu/npu_base_layer.h"
#include "nlohmann/json.hpp"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/base_operation.h"

namespace xllm::kernel {

class NpuAttentionImpl : public xllm::layer::NpuBaseLayer {
 public:
  explicit NpuAttentionImpl(const std::string& opType,
                            const std::string& opParam,
                            const std::string& opName);

  ~NpuAttentionImpl() {};

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string weight_str) const;

  void merge_loaded_weights();

  void forward(torch::Tensor& intermediate_k,
               torch::Tensor& intermediate_v,
               torch::Tensor& in_k_cache,
               torch::Tensor& in_v_cache,
               torch::Tensor& in_slots,
               int nodeId);

 private:
  int64_t init_node(atb_speed::Model::Node& node,
                    const std::string& opType,
                    const std::string& opParam,
                    const std::string& opName);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& intermediate_k,
                               torch::Tensor& intermediate_v,
                               torch::Tensor& in_k_cache,
                               torch::Tensor& in_v_cache,
                               torch::Tensor& in_slots);

  // void param_from_args(atb::infer::SelfAttentionParam& param, const
  // ModelArgs& args, int32_t numAttentionHeadsPerRank, int32_t
  // hiddenSizePerAttentionHead);

  atb_speed::Model::Node attention_node_;
  std::string model_name_;
  atb::infer::SelfAttentionParam attention_param_;
  atb::Tensor internal_intermediate_k;
  atb::Tensor internal_intermediate_v;
  atb::Tensor internal_in_k_cache;
  atb::Tensor internal_in_v_cache;
  atb::Tensor internal_in_slots;
};

}  // namespace xllm::kernel
