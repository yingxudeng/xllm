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
#include "npu/npu_attention_impl.h"
#include "npu_v1/ops_npu/npu_ops_api.h"
#endif

namespace xllm::kernel {
#if defined(USE_NPU)
class Attention : public torch::nn::ModuleHolder<NpuAttentionImpl> {
 public:
  using torch::nn::ModuleHolder<NpuAttentionImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuAttentionImpl;

  //   Attention(const ModelContext& context, const std::string &opType, const
  //   std::string &opParam, const std::string &opName)
  //       : ModuleHolder(std::make_shared<NpuAttentionImpl>(context, opType,
  //       opParam, opName)) {}

  void npu_paged_attention(const at::Tensor& query,
                           const at::Tensor& key_cache,
                           const at::Tensor& value_cache,
                           int64_t num_kv_heads,
                           int64_t num_heads,
                           double scale_value,
                           const at::Tensor& block_table,
                           const at::Tensor& context_lens,
                           at::Tensor& out) {
    _npu_paged_attention(query,
                         key_cache,
                         value_cache,
                         num_kv_heads,
                         num_heads,
                         scale_value,
                         block_table,
                         context_lens,
                         out);
  }
};
#endif
}  // namespace xllm::kernel
