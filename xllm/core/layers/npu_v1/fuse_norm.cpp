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

#include "fuse_norm.h"

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "kernels/mlu/torch_ops_api.h"

namespace xllm {
namespace layer {

void print_first_5_1(const torch::Tensor& tensor, const std::string& name) {
  try {
    std::stringstream ss;
    auto float_tensor = tensor.to(torch::kCPU, torch::kFloat);
    auto flattened = float_tensor.flatten();
    int64_t size = flattened.size(0);

    ss << std::setprecision(4) << std::fixed;
    ss << name << ":\n";

    ss << "  Shape: " << tensor.sizes() << "\n";
    ss << "  Data Type: " << tensor.dtype() << "\n";
    ss << "  Device: " << tensor.device() << "\n";

    // 打印前4个元素
    ss << "  First 4 elements: ";
    for (int i = 0; i < std::min(static_cast<int64_t>(4), size); ++i) {
      ss << flattened[i].item<float>() << " ";
    }
    ss << "\n";

    // 打印后4个元素
    ss << "  Last 4 elements:  ";
    for (int i = std::max(static_cast<int64_t>(0), size - 4); i < size; ++i) {
      ss << flattened[i].item<float>() << " ";
    }
    ss << "\n";

    // 计算统计值
    if (size > 0) {
      float max_val = flattened.max().item<float>();
      float min_val = flattened.min().item<float>();
      float mean_val = flattened.mean().item<float>();

      ss << std::fixed << std::setprecision(8);
      ss << "  Max: " << max_val << "\n";
      ss << "  Min: " << min_val << "\n";
      ss << "  Mean: " << mean_val << "\n";
    } else {
      ss << "  Tensor is empty\n";
    }
    ss << std::endl;

    std::cout << ss.str();
    std::cout.flush();
  } catch (const c10::Error& e) {
    std::cerr << "PyTorch Error in print_first_5_1: " << e.what() << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << "Runtime Error in print_first_5_1: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Standard Exception in print_first_5_1: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "Unknown Error in print_first_5_1" << std::endl;
  }
}

const static std::string kLayerNormMode = "layernorm";
const static std::string kRmsNormMode = "rmsnorm";

FusedRMSNormImpl::FusedRMSNormImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight",
                               torch::empty({dim}, options),
                               /*requires_grad=*/false);
}

torch::Tensor FusedRMSNormImpl::forward(torch::Tensor& input) {
  // auto org_shape = input.sizes().vec();
  input = input.reshape({-1, norm_dim_});
  // auto output = torch::empty_like(input);

  // tmo::torch_api::fused_layernorm(input,
  //                                 output,
  //                                 std::nullopt /*residual*/,
  //                                 weight_,
  //                                 std::nullopt /*beta*/,
  //                                 std::nullopt /*bias*/,
  //                                 std::nullopt /*quant_scale*/,
  //                                 std::nullopt /*residual_out*/,
  //                                 std::nullopt /*smooth_quant_scale*/,
  //                                 std::nullopt /*normed_out*/,
  //                                 kRmsNormMode,
  //                                 eps_,
  //                                 false /*store_output_before_norm*/,
  //                                 false /*store_output_after_norm*/,
  //                                 false /*dynamic_quant*/
  // );

  // output = output.view(org_shape);
  std::cerr << "dyx-debug FusedRMSNormImpl::forward input shape before "
               "npu_rms_norm. input.sizes(): "
            << input.sizes() << " weight_ shape: " << weight_.sizes()
            << " eps_: " << eps_ << std::endl;
  // sleep(1);
  print_first_5_1(input, "FusedRMSNormImpl::forward input before npu_rms_norm");
  print_first_5_1(weight_,
                  "FusedRMSNormImpl::forward weight_ before npu_rms_norm");
  std::cerr << "eps_: " << eps_ << std::endl;
  std::tuple<at::Tensor, at::Tensor> result =
      at_npu::native::custom_ops::npu_rms_norm(input, weight_, eps_);
  // std::tuple<at::Tensor, at::Tensor> result =
  // at_npu::native::custom_ops::npu_rms_norm(input, 1.0 + weight_, eps_);
  at::Tensor normalized_input = std::get<0>(result);
  // input, _ = at_npu::native::custom_ops::npu_rms_norm(input, 1.0 + weight_,
  //                                     eps_);
  std::cerr << "dyx-debug FusedRMSNormImpl::forward normalized_input shape "
               "after npu_rms_norm. normalized_input.sizes(): "
            << normalized_input.sizes() << std::endl;
  print_first_5_1(
      normalized_input,
      "FusedRMSNormImpl::forward normalized_input after npu_rms_norm");
  return normalized_input;
}

void FusedRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes())
        << "weight size mismatch for " << name();
    weight_.copy_(weight);
  }
}

}  // namespace layer
}  // namespace xllm
