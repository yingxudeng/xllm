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

#include <glog/logging.h>
#include <torch/torch.h>

#if defined(USE_MLU)
#include "mlu/linear_impl.h"
#endif

#if defined(USE_NPU)
#include "npu_v1/linear_impl.h"
#endif
namespace xllm {
namespace layer {

#if defined(USE_MLU)
class ColumnParallelLinear
    : public torch::nn::ModuleHolder<ColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ColumnParallelLinearImpl;

  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<ColumnParallelLinearImpl>(in_features,
                                                                out_features,
                                                                bias,
                                                                gather_output,
                                                                parallel_args,
                                                                options)) {}
};

class QKVParallelLinear
    : public torch::nn::ModuleHolder<QKVParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<QKVParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = QKVParallelLinearImpl;

  QKVParallelLinear(int64_t hidden_size,
                    int64_t num_heads,
                    int64_t num_kv_heads,
                    int64_t head_size,
                    int64_t num_kv_head_replicas,
                    bool bias,
                    bool gather_output,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options)
      : ModuleHolder(
            std::make_shared<QKVParallelLinearImpl>(hidden_size,
                                                    num_heads,
                                                    num_kv_heads,
                                                    head_size,
                                                    num_kv_head_replicas,
                                                    bias,
                                                    gather_output,
                                                    parallel_args,
                                                    options)) {}
};

class RowParallelLinear
    : public torch::nn::ModuleHolder<RowParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<RowParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RowParallelLinearImpl;

  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool bias,
                    bool input_is_parallelized,
                    bool if_reduce_results,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options)
      : ModuleHolder(
            std::make_shared<RowParallelLinearImpl>(in_features,
                                                    out_features,
                                                    bias,
                                                    input_is_parallelized,
                                                    if_reduce_results,
                                                    parallel_args,
                                                    options)) {}
};

class ReplicatedLinear : public torch::nn::ModuleHolder<ReplicatedLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ReplicatedLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ReplicatedLinearImpl;

  ReplicatedLinear(int64_t in_features,
                   int64_t out_features,
                   bool bias,
                   const QuantArgs& quant_args,
                   const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<ReplicatedLinearImpl>(in_features,
                                                            out_features,
                                                            bias,
                                                            quant_args,
                                                            options)) {}
};
#endif

#if defined(USE_NPU)
class ColumnParallelLinearV1
    : public torch::nn::ModuleHolder<ColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ColumnParallelLinearImpl;

  ColumnParallelLinearV1(int64_t in_features,
                         int64_t out_features,
                         bool bias,
                         bool gather_output,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<ColumnParallelLinearImpl>(in_features,
                                                                out_features,
                                                                bias,
                                                                gather_output,
                                                                parallel_args,
                                                                options)) {}
};

class QKVParallelLinear
    : public torch::nn::ModuleHolder<QKVParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<QKVParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = QKVParallelLinearImpl;

  QKVParallelLinear(int64_t hidden_size,
                    int64_t num_heads,
                    int64_t num_kv_heads,
                    int64_t head_size,
                    int64_t num_kv_head_replicas,
                    bool bias,
                    bool gather_output,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options)
      : ModuleHolder(
            std::make_shared<QKVParallelLinearImpl>(hidden_size,
                                                    num_heads,
                                                    num_kv_heads,
                                                    head_size,
                                                    num_kv_head_replicas,
                                                    bias,
                                                    gather_output,
                                                    parallel_args,
                                                    options)) {}
};

class RowParallelLinear
    : public torch::nn::ModuleHolder<RowParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<RowParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RowParallelLinearImpl;

  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool bias,
                    bool input_is_parallelized,
                    bool if_reduce_results,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options)
      : ModuleHolder(
            std::make_shared<RowParallelLinearImpl>(in_features,
                                                    out_features,
                                                    bias,
                                                    input_is_parallelized,
                                                    if_reduce_results,
                                                    parallel_args,
                                                    options)) {}
};

class ReplicatedLinear : public torch::nn::ModuleHolder<ReplicatedLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ReplicatedLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ReplicatedLinearImpl;

  ReplicatedLinear(int64_t in_features,
                   int64_t out_features,
                   bool bias,
                   const QuantArgs& quant_args,
                   const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<ReplicatedLinearImpl>(in_features,
                                                            out_features,
                                                            bias,
                                                            quant_args,
                                                            options)) {}
};
#endif

}  // namespace layer
}  // namespace xllm
