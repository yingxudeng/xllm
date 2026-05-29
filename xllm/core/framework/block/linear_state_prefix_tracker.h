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

#include <unordered_set>
#include <vector>

#include "framework/block/block.h"
#include "util/hash_util.h"

namespace xllm {

class LinearStatePrefixTracker {
 public:
  LinearStatePrefixTracker(int32_t dp_size, bool enabled);

  void add_checkpoints(int32_t dp_rank, const std::vector<XXH3Key>& hashes);
  bool has_checkpoint(int32_t dp_rank, const XXH3Key& hash) const;
  void remove_checkpoints(int32_t dp_rank, const std::vector<XXH3Key>& hashes);

  void evict_checkpoints(int32_t dp_rank,
                         const std::vector<XXH3Key>& evicted_hashes);
  std::vector<PrefixHash> drain_evictions();

  size_t find_safe_prefix_length(int32_t dp_rank,
                                 const std::vector<Block>& shared_blocks,
                                 size_t existed_shared_blocks_num) const;

  bool enabled() const { return enabled_; }

 private:
  using CheckpointHashSet =
      std::unordered_set<XXH3Key, FixedStringKeyHash, FixedStringKeyEqual>;

  // Validates dp_rank and returns the checkpoint hash set for that rank.
  CheckpointHashSet& checkpoint_set(int32_t dp_rank);
  const CheckpointHashSet& checkpoint_set(int32_t dp_rank) const;

  bool enabled_;
  std::vector<CheckpointHashSet> checkpoint_hashes_;
  std::vector<PrefixHash> pending_evictions_;
};

}  // namespace xllm
