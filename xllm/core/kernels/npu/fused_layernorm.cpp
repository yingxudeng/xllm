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
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps,
                       const std::string& mode) {
  if (mode != "rmsnorm") {
    LOG(FATAL) << "Only rmsnorm mode is supported in NPU rms_norm";
  }
  std::tuple<at::Tensor, at::Tensor> result =
      at_npu::native::custom_ops::npu_rms_norm(input, weight, eps);
  auto normalized_input = std::get<0>(result);
  return normalized_input;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon) {
  return at_npu::native::custom_ops::npu_add_rms_norm(x1, x2, gamma, epsilon);
}

}  // namespace xllm::kernel::npu