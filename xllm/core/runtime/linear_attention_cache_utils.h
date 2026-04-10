/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>

namespace xllm {

inline torch::ScalarType get_linear_attention_ssm_cache_dtype(
    torch::ScalarType model_dtype) {
  static_cast<void>(model_dtype);
  // Keep recurrent SSM state in FP32 to avoid per-step precision loss.
  return torch::kFloat32;
}

inline int64_t get_linear_attention_ssm_cache_dtype_size_in_bytes(
    int64_t model_dtype_size) {
  static_cast<void>(model_dtype_size);
  return static_cast<int64_t>(sizeof(float));
}

}  // namespace xllm
