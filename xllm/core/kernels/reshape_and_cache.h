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
#include "npu/npu_reshape_and_cache_impl.h"
#endif

namespace xllm::kernel {
#if defined(USE_NPU)
class ReshapeAndCache : public torch::nn::ModuleHolder<NpuReshapeAndCacheImpl> {
 public:
  using torch::nn::ModuleHolder<NpuReshapeAndCacheImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuReshapeAndCacheImpl;

  ReshapeAndCache(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuReshapeAndCacheImpl>(context)) {}
};
#endif
}  // namespace xllm::kernel
