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

#include "framework/block/linear_state_prefix_cache.h"

#include <glog/logging.h>

#include <iterator>
#include <unordered_set>
#include <utility>

namespace xllm {

LinearStatePrefixCache::LinearStatePrefixCache(int32_t num_slots)
    : num_slots_(num_slots),
      slots_(static_cast<uint32_t>(num_slots),
             /*resource_name=*/"linear state slot",
             /*exhaustion_message=*/"") {
  CHECK_GT(num_slots, 0) << "linear state prefix cache needs at least one slot";
}

bool LinearStatePrefixCache::ensure_free_slot(
    const std::unordered_set<int32_t>& protected_slots) {
  if (slots_.num_free_blocks() > 0) {
    return true;
  }
  if (lru_.empty()) {
    // Every slot is live; nothing can be reclaimed.
    return false;
  }

  auto victim_it = lru_.begin();
  while (victim_it != lru_.end()) {
    auto cached_it = cached_slots_.find(*victim_it);
    CHECK(cached_it != cached_slots_.end())
        << "LRU and linear-state prefix cache out of sync";
    if (protected_slots.find(cached_it->second.block.id()) ==
        protected_slots.end()) {
      // Erasing the entry drops its Block handle, which returns the slot to the
      // underlying free list.
      lru_.erase(victim_it);
      cached_slots_.erase(cached_it);
      return slots_.num_free_blocks() > 0;
    }
    ++victim_it;
  }

  return false;
}

Block LinearStatePrefixCache::acquire_live_slot() {
  if (!ensure_free_slot()) {
    return Block();
  }
  auto blocks = slots_.allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear state slot allocation failed";
  return std::move(blocks[0]);
}

int32_t LinearStatePrefixCache::match(const XXH3Key& restore_hash) {
  auto it = cached_slots_.find(restore_hash);
  if (it == cached_slots_.end()) {
    return -1;
  }
  touch(restore_hash, it->second);
  return it->second.block.id();
}

int32_t LinearStatePrefixCache::insert(const XXH3Key& save_hash) {
  const std::unordered_set<int32_t> protected_slots;
  return insert(save_hash, protected_slots);
}

int32_t LinearStatePrefixCache::insert(
    const XXH3Key& save_hash,
    const std::unordered_set<int32_t>& protected_slots) {
  auto it = cached_slots_.find(save_hash);
  if (it != cached_slots_.end()) {
    // Already cached: no new slot and no copy are needed.
    touch(save_hash, it->second);
    return it->second.block.id();
  }
  if (!ensure_free_slot(protected_slots)) {
    return -1;
  }
  auto blocks = slots_.allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear state checkpoint allocation failed";
  const int32_t slot_id = blocks[0].id();
  lru_.push_back(save_hash);
  cached_slots_.emplace(
      save_hash, CacheEntry{std::move(blocks[0]), std::prev(lru_.end())});
  return slot_id;
}

bool LinearStatePrefixCache::contains(const XXH3Key& prefix_hash) const {
  return cached_slots_.find(prefix_hash) != cached_slots_.end();
}

void LinearStatePrefixCache::touch(const XXH3Key& prefix_hash,
                                   CacheEntry& entry) {
  lru_.erase(entry.lru_it);
  lru_.push_back(prefix_hash);
  entry.lru_it = std::prev(lru_.end());
}

}  // namespace xllm
