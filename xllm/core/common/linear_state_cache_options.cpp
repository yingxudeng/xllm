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

#include <boost/algorithm/string.hpp>
#include <cmath>

namespace xllm {

LinearStateCachePolicy parse_linear_state_cache_policy(
    const std::string& policy) {
  const std::string normalized = boost::algorithm::to_lower_copy(policy);
  if (normalized == "compat") {
    return LinearStateCachePolicy::COMPAT;
  }
  if (normalized == "fixed") {
    return LinearStateCachePolicy::FIXED;
  }
  if (normalized == "auto") {
    return LinearStateCachePolicy::AUTO;
  }
  LOG(FATAL) << "Unsupported linear_state_cache_policy: " << policy
             << ". Supported values are: compat, fixed, auto.";
  return LinearStateCachePolicy::COMPAT;
}

const char* linear_state_cache_policy_to_string(LinearStateCachePolicy policy) {
  switch (policy) {
    case LinearStateCachePolicy::COMPAT:
      return "compat";
    case LinearStateCachePolicy::FIXED:
      return "fixed";
    case LinearStateCachePolicy::AUTO:
      return "auto";
  }
  return "unknown";
}

void validate_linear_state_cache_options(
    const LinearStateCacheOptions& options) {
  if (options.policy() == LinearStateCachePolicy::FIXED) {
    CHECK_GT(options.max_linear_state_cache_slots(), 0)
        << "max_linear_state_cache_slots must be greater than 0 when "
           "linear_state_cache_policy=fixed.";
  }
  const double full_kv_memory_ratio =
      options.linear_state_full_kv_memory_ratio();
  CHECK(std::isfinite(full_kv_memory_ratio))
      << "linear_state_full_kv_memory_ratio must be finite.";
  CHECK_GT(full_kv_memory_ratio, 0.0)
      << "linear_state_full_kv_memory_ratio must be greater than 0.";
  CHECK_GE(options.min_full_kv_cache_blocks(), 0)
      << "min_full_kv_cache_blocks must be greater than or equal to 0.";
}

}  // namespace xllm
