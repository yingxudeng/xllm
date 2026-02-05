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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const std::optional<torch::Tensor>& positions,
                  const torch::Tensor& cos_cache,
                  const torch::Tensor& sin_cache,
                  bool use_precomputed_cos_sin) {
  torch::Tensor cos;
  torch::Tensor sin;
  int64_t rotary_dim = 0;
  if (use_precomputed_cos_sin) {
    cos = cos_cache;
    sin = sin_cache;
    rotary_dim = cos.size(-1);
  } else {
    TORCH_CHECK(positions.has_value(),
                "positions must be provided when use_precomputed_cos_sin is "
                "false");
    auto cos_sin = cos_sin_cache.index_select(0, positions.value());
    int64_t last_dim = cos_sin.size(-1);
    rotary_dim = last_dim / 2;
    auto cos_sin_split = cos_sin.chunk(2, /*dim=*/-1);
    cos = cos_sin_split[0];
    sin = cos_sin_split[1];
  }

  // Ensure tensors are contiguous for NPU operations
  cos = cos.contiguous().view({1, -1, 1, rotary_dim});
  sin = sin.contiguous().view({1, -1, 1, rotary_dim});

  q = q.view({1, q.size(0), -1, rotary_dim});
  k = k.view({1, k.size(0), -1, rotary_dim});

  at_npu::native::custom_ops::npu_apply_rotary_pos_emb(q, k, cos, sin);
}

}  // namespace xllm::kernel::npu
