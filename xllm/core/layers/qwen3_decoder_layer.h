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
// #include "npu/npu_qwen3_decoder_layer_impl.h"
#include "npu_v1/qwen3_decoder_layer.h"
#elif defined(USE_MLU)
#include "mlu/qwen3_decoder_layer.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
// class Qwen3DecoderLayer
//     : public torch::nn::ModuleHolder<NpuQwen3DecoderLayerImpl> {
//  public:
//   using torch::nn::ModuleHolder<NpuQwen3DecoderLayerImpl>::ModuleHolder;
//   using Impl __attribute__((__unused__)) = NpuQwen3DecoderLayerImpl;

//   Qwen3DecoderLayer(const ModelContext& context)
//       : ModuleHolder(std::make_shared<NpuQwen3DecoderLayerImpl>(context)) {}
// };
class Qwen3DecoderLayer : public torch::nn::ModuleHolder<Qwen3DecoderImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3DecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3DecoderImpl;

  Qwen3DecoderLayer(const ModelContext& context)
      : ModuleHolder(std::make_shared<Qwen3DecoderImpl>(context)) {}
};
#elif defined(USE_MLU)
class Qwen3DecoderLayer : public torch::nn::ModuleHolder<Qwen3DecoderImpl> {
 public:
  using torch::nn::ModuleHolder<Qwen3DecoderImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = Qwen3DecoderImpl;

  Qwen3DecoderLayer(const ModelContext& context)
      : ModuleHolder(std::make_shared<Qwen3DecoderImpl>(context)) {}
};
#endif

}  // namespace layer
}  // namespace xllm
