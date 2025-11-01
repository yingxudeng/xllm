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

#if defined(USE_NPU)
#include "npu/npu_lm_head_impl.h"
#include "npu_v1/linear_impl.h"

#elif defined(USE_MLU)
#include "mlu/linear_impl.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class LmHead : public torch::nn::ModuleHolder<NpuLmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<NpuLmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuLmHeadImpl;

  LmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuLmHeadImpl>(context)) {}
};

class LmHeadV1 : public torch::nn::ModuleHolder<ColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ColumnParallelLinearImpl;

  LmHeadV1(int64_t in_features,
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

#elif defined(USE_MLU)
class LmHead : public torch::nn::ModuleHolder<ColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ColumnParallelLinearImpl;

  LmHead(int64_t in_features,
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
#endif

}  // namespace layer
}  // namespace xllm