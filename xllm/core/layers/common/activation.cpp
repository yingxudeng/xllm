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

#include "activation.h"

#include <glog/logging.h>

#include <cmath>

#include "kernels/ops_api.h"
#include "platform/device.h"

namespace {

bool should_use_npu_activation_fallback(const std::string& act_mode,
                                        bool is_gated) {
  if (xllm::Device::type_str() != "npu") {
    return false;
  }

  // The current NPU activation kernel only provides the gated swiglu path.
  return !is_gated ||
         (act_mode != xllm::kernel::kActModeSilu && act_mode != "swiglu");
}

torch::Tensor apply_torch_activation(const torch::Tensor& input,
                                     const std::string& act_mode) {
  if (act_mode == xllm::kernel::kActModeSilu ||
      act_mode == xllm::kernel::kActModeSwish) {
    return input * torch::sigmoid(input);
  }

  if (act_mode == xllm::kernel::kActModeGelu) {
    return 0.5 * input * (1.0 + torch::erf(input / std::sqrt(2.0)));
  }

  if (act_mode == "gelu_tanh" || act_mode == "gelu_pytorch_tanh") {
    constexpr double kGeluTanhCoeff = 0.7978845608028654;  // sqrt(2 / pi)
    constexpr double kGeluTanhCubic = 0.044715;
    auto input_cubed = input * input * input;
    return 0.5 * input *
           (1.0 + torch::tanh(kGeluTanhCoeff *
                              (input + kGeluTanhCubic * input_cubed)));
  }

  if (act_mode == xllm::kernel::kActModeQuickGelu) {
    return input * torch::sigmoid(1.702 * input);
  }

  LOG(FATAL) << "Unsupported NPU activation fallback mode: " << act_mode;
}

torch::Tensor active_with_torch_fallback(const torch::Tensor& input,
                                         const std::string& act_mode,
                                         bool is_gated) {
  if (!is_gated) {
    return apply_torch_activation(input, act_mode);
  }

  CHECK_EQ(input.size(-1) % 2, 0)
      << "Gated activation expects an even hidden size, got " << input.size(-1);
  auto chunks = input.chunk(2, /*dim=*/-1);
  return apply_torch_activation(chunks[0], act_mode) * chunks[1];
}

}  // namespace

namespace xllm {
namespace layer {

ActivationImpl::ActivationImpl(const std::string& act_mode, bool is_gated)
    : act_mode_(act_mode), is_gated_(is_gated) {}

void ActivationImpl::forward(torch::Tensor& input, torch::Tensor& output) {
  if (should_use_npu_activation_fallback(act_mode_, is_gated_)) {
    output = active_with_torch_fallback(input, act_mode_, is_gated_);
    return;
  }

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = input;
  activation_params.output = output;
  activation_params.act_mode = act_mode_;
  activation_params.is_gated = is_gated_;
  xllm::kernel::active(activation_params);
  // Unified assignment: NPU returns new tensor, others modify in-place (no-op
  // assignment)
  output = activation_params.output;
}

}  // namespace layer
}  // namespace xllm
