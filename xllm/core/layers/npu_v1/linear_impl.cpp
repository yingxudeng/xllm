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

#include "linear_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>
// #include <torch/distributed.h>
// #include "c10d/ProcessGroup.hpp"
// #include
// "/usr/local/lib64/python3.11/site-packages/torch/include/torch/csrc/distributed/c10d/ProcessGroup.hpp"
// #include
// "/usr/local/lib64/python3.11/site-packages/torch/include/torch/csrc/api/include/torch/data/samplers/distributed.h"

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/mlu/torch_ops_api.h"

namespace xllm {
namespace layer {

void print_first_5_3(const torch::Tensor& tensor, const std::string& name) {
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

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : gather_output_(gather_output),
      parallel_args_(parallel_args),
      device_(options.device()) {
  rank_ = parallel_args_.rank();
  world_size_ = parallel_args_.world_size();
  CHECK(out_features % world_size_ == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size_;
  const int64_t out_features_per_partition = out_features / world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, in_features}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);

  print_first_5_3(input, "ColumnParallelLinear input_");
  torch::Tensor output;
  if (bias_.defined() && rank_ == 0) {
    output = torch::nn::functional::linear(input, weight_, bias_);
  } else {
    output = torch::nn::functional::linear(input, weight_);
  }
  print_first_5_3(output, "ColumnParallelLinear output");
  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto rank = rank_;
  const auto world_size = world_size_;

  // load sharded weights on dim 0
  LOAD_SHARDED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    // load sharded bias on dim 0
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const auto rank = rank_;
  const auto world_size = world_size_;

  // load and merge the weights on dim 0
  LOAD_FUSED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    // load and merge the bias on dim 0
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

QKVParallelLinearImpl::QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()) {
  const int32_t QKV_CNT = 3;
  rank_ = parallel_args_.rank();
  world_size_ = parallel_args_.world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  qkv_weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, hidden_size}, options),
      /*requires_grad=*/false);
  qkv_weight_list_.resize(QKV_CNT);

  if (bias) {
    qkv_bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
    qkv_bias_list_.resize(QKV_CNT);
  }
}

torch::Tensor QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  at::Tensor output;
  print_first_5_3(input, "QKVParallelLinearImpl::forward input");
  if (qkv_bias_.defined() && rank_ == 0) {
    output = torch::nn::functional::linear(input, qkv_weight_, qkv_bias_);
  } else {
    output = torch::nn::functional::linear(input, qkv_weight_);
  }
  print_first_5_3(output,
                  "QKVParallelLinearImpl::forward output before gather");
  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

bool QKVParallelLinearImpl::load_qkv_weight(const StateDict& state_dict,
                                            int32_t index) {
  if (qkv_weight_list_[index].defined() || state_dict.size() == 0) {
    return false;
  }
  DEFINE_WEIGHT(weight);
  int64_t out_feature = num_heads_ * head_size_;
  int32_t rank = rank_;
  int world_size = world_size_;
  if (index > 0) {
    rank = rank_ / num_kv_head_replicas_;
    world_size = world_size_ / num_kv_head_replicas_;
    out_feature = num_kv_heads_ * head_size_;
  }
  weight_ = torch::empty({out_feature, hidden_size_}, options_);
  LOAD_SHARDED_WEIGHT(weight, 0);
  if (weight_is_loaded_) {
    qkv_weight_list_[index] = weight_.clone();
  }
  return weight_is_loaded_;
}

void QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  std::vector<std::string> prefixes = {"q_proj.", "k_proj.", "v_proj."};
  if (!qkv_weight_is_loaded_) {
    bool all_loaded = true;
    for (size_t i = 0; i < prefixes.size(); ++i) {
      all_loaded =
          all_loaded &&
          load_qkv_weight(state_dict.get_dict_with_prefix(prefixes[i]), i);
    }
    if (all_loaded) {
      const auto merged_weight = torch::cat(qkv_weight_list_, /*dim=*/0);
      CHECK_EQ(qkv_weight_.sizes(), merged_weight.sizes())
          << "weight size mismatch";
      qkv_weight_.copy_(merged_weight);
      // release the memory for weight_list
      qkv_weight_list_.clear();
      qkv_weight_is_loaded_ = true;
    }
  }
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool if_reduce_results,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : input_is_parallelized_(input_is_parallelized),
      if_reduce_results_(if_reduce_results),
      parallel_args_(parallel_args) {
  rank_ = parallel_args_.rank();
  world_size_ = parallel_args_.world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features, in_features_per_partition}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  if (!input_is_parallelized_) {
    // input = xllm::parallel_state::scatter(input, parallel_args_.tp_group_);
    // torch::distributed::scatter(input, parallel_args_.tp_group_);
  }

  at::Tensor output;
  if (bias_.defined() && rank_ == 0) {
    output = torch::nn::functional::linear(input, weight_, bias_);
  } else {
    output = torch::nn::functional::linear(input, weight_);
  }
  if (if_reduce_results_ && world_size_ > 1) {
    // output = xllm::parallel_state::reduce(output, parallel_args_.tp_group_);
    // torch::distributed::all_reduce(output, parallel_args_.tp_group_);
    std::cerr << "RowParallelLinearImpl::  will allreduce" << std::endl;
    std::cerr << "process_group_: " << parallel_args_.process_group_
              << std::endl;
    // parallel_args_.process_group_->allreduce(output);
    // parallel_args_.process_group_->allreduce(input);
    output = xllm::parallel_state::reduce(output, parallel_args_.tp_group_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto rank = rank_;
  const auto world_size = world_size_;

  // load sharded weights on dim 1
  LOAD_SHARDED_WEIGHT(weight, 1);

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options) {
  weight_ =
      register_parameter("weight",
                         torch::empty({out_features, in_features}, options),
                         /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = bias_.defined() ? F::linear(input, weight_, bias_)
                                : F::linear(input, weight_);
  return output;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm
