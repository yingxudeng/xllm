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

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"
namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         torch::Tensor& value,
                         torch::Tensor& k_cache,
                         torch::Tensor& v_cache,
                         const torch::Tensor& slot_mapping) {
  atb::_npu_reshape_and_cache(key, value, k_cache, v_cache, slot_mapping);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   int num_heads,
                   int num_kv_heads,
                   torch::Tensor& output) {
  atb::_npu_flash_attention(
      query, key, value, mask, seq_len, scale, num_heads, num_kv_heads, output);
}

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  int num_kv_heads,
                  int num_heads,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output) {
  atb::_npu_paged_attention(query,
                            k_cache,
                            v_cache,
                            num_kv_heads,
                            num_heads,
                            scale,
                            block_table,
                            seq_lens,
                            output);
}

}  // namespace xllm::kernel::npu