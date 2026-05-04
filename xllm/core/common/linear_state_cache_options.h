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

#include <cstdint>

#include "common/macros.h"

namespace xllm {

struct LinearStateCacheOptions {
  // Active linear-state slots. 0 derives capacity from the memory ratio.
  PROPERTY(int64_t, max_linear_state_cache_slots) = 0;

  // Ratio of linear-state memory to full-attention KV memory in auto mode.
  PROPERTY(double, linear_state_full_kv_memory_ratio) = 0.9;
};

void validate_linear_state_cache_options(
    const LinearStateCacheOptions& options);

}  // namespace xllm
