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

#include "qwen3_decoder_layer.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

void print_first_5(const torch::Tensor& tensor, const std::string& name) {
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
    std::cerr << "PyTorch Error in print_first_5: " << e.what() << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << "Runtime Error in print_first_5: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Standard Exception in print_first_5: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "Unknown Error in print_first_5" << std::endl;
  }
}

Qwen3DecoderImpl::Qwen3DecoderImpl(const ModelContext& context) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  // Initialize attention layers
  attention_ = register_module(
      "self_attn",
      Qwen3Attention(model_args, quant_args, parallel_args, options, context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RmsNormV1(model_args.hidden_size(), model_args.rms_norm_eps(), options));
  // input_norm_ = register_module("input_layernorm",
  // xllm::kernel::RmsNorm(context));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RmsNormV1(model_args.hidden_size(), model_args.rms_norm_eps(), options));
  // post_norm_ = register_module("post_attention_layernorm",
  // xllm::kernel::RmsNorm(context));

  // Initialize mlp
  mlp_ = register_module("mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  true,
                                  false,
                                  quant_args,
                                  parallel_args,
                                  options));
}

void Qwen3DecoderImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  std::cerr << "dyx-debug Loading Input Norm Weights" << std::endl;
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  std::cerr << "dyx-debug Loading Post Norm Weights" << std::endl;
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

torch::Tensor Qwen3DecoderImpl::forward(torch::Tensor& x,
                                        torch::Tensor& positions,
                                        const AttentionMetadata& attn_metadata,
                                        KVCache& kv_cache,
                                        const ModelInputParams& input_params) {
  // Pre-attention norm
  std::cout << "dyx-debug: [Pre-Norm] x shape: " << x.sizes() << std::endl;
  auto residual = x;
  std::cout << "dyx-debug: [Residual] residual shape: " << residual.sizes()
            << std::endl;

  print_first_5(x, "before input_norm x");
  x = input_norm_->forward(x);
  std::cout << "dyx-debug: [Post-Norm] x shape after input_norm: " << x.sizes()
            << std::endl;
  print_first_5(x, "after input_norm and before attention x");
  // Attention
  std::cout << "dyx-debug: [Pre-Attention] x shape: " << x.sizes()
            << ", positions shape: " << positions.sizes() << std::endl;
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  std::cout << "dyx-debug: [Post-Attention] x shape after attention: "
            << x.sizes() << std::endl;
  print_first_5(x, "after attention x");

  std::cout << "dyx-debug: [Pre-Residual-Add] x shape: " << x.sizes()
            << ", residual shape: " << residual.sizes() << std::endl;
  x = x + residual;
  // std::cout << "dyx-debug: [Post-Residual-Add] x shape after residual add: "
  // << x.sizes() << std::endl; print_first_5(x, "after residual add x");
  // Post-attention norm
  residual = x;
  x = post_norm_->forward(x);
  print_first_5(x, "after post_norm x");

  // MLP forward
  x = mlp_->forward(x);
  print_first_5(x, "after mlp x");
  x = x + residual;

  return x;
}

}  // namespace layer
}  // namespace xllm
