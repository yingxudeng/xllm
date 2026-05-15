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

#include "linear_state_prefix_tracker.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

namespace xllm {

LinearStatePrefixTracker::LinearStatePrefixTracker(int32_t dp_size,
                                                   bool enabled)
    : enabled_(enabled) {
  if (enabled_) {
    checkpoint_hashes_.resize(dp_size);
  }
}

void LinearStatePrefixTracker::add_checkpoints(
    int32_t dp_rank,
    const std::vector<XXH3Key>& hashes) {
  if (!enabled_ || hashes.empty()) {
    return;
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), checkpoint_hashes_.size());
  auto& set = checkpoint_hashes_[dp_rank];
  for (const XXH3Key& hash : hashes) {
    set.insert(hash);
  }
}

bool LinearStatePrefixTracker::has_checkpoint(int32_t dp_rank,
                                              const XXH3Key& hash) const {
  if (!enabled_) {
    return false;
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), checkpoint_hashes_.size());
  return checkpoint_hashes_[dp_rank].find(hash) !=
         checkpoint_hashes_[dp_rank].end();
}

void LinearStatePrefixTracker::remove_checkpoints(
    int32_t dp_rank,
    const std::vector<XXH3Key>& hashes) {
  if (!enabled_ || hashes.empty()) {
    return;
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), checkpoint_hashes_.size());
  auto& set = checkpoint_hashes_[dp_rank];
  for (const XXH3Key& hash : hashes) {
    set.erase(hash);
  }
}

void LinearStatePrefixTracker::evict_checkpoints(
    int32_t dp_rank,
    const std::vector<XXH3Key>& evicted_hashes) {
  if (!enabled_ || evicted_hashes.empty()) {
    return;
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), checkpoint_hashes_.size());
  auto& set = checkpoint_hashes_[dp_rank];
  for (const auto& hash : evicted_hashes) {
    if (set.erase(hash) > 0) {
      pending_evictions_.emplace_back(to_prefix_hash(hash));
    }
  }
}

std::vector<PrefixHash> LinearStatePrefixTracker::drain_evictions() {
  return std::exchange(pending_evictions_, {});
}

size_t LinearStatePrefixTracker::find_safe_prefix_length(
    int32_t dp_rank,
    const std::vector<Block>& shared_blocks,
    size_t existed_shared_blocks_num) const {
  if (!enabled_) {
    return shared_blocks.size();
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), checkpoint_hashes_.size());

  size_t safe_blocks =
      std::min(existed_shared_blocks_num, shared_blocks.size());
  for (size_t block_idx = safe_blocks; block_idx < shared_blocks.size();
       ++block_idx) {
    const XXH3Key prefix_hash(
        shared_blocks[block_idx].get_immutable_hash_value());
    if (has_checkpoint(dp_rank, prefix_hash)) {
      safe_blocks = block_idx + 1;
    }
  }
  return safe_blocks;
}

}  // namespace xllm
