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

#include "common/linear_state_cache_options.h"

#include <glog/logging.h>

namespace xllm {

void validate_linear_state_cache_options(
    const LinearStateCacheOptions& options) {
  CHECK_GE(options.max_linear_state_cache_slots(), 0)
      << "max_linear_state_cache_slots must be greater than or equal to 0.";
}

}  // namespace xllm
