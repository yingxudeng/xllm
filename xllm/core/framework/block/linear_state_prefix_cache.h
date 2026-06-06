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
#include <list>
#include <unordered_map>
#include <unordered_set>

#include "framework/block/block.h"
#include "framework/block/single_block_manager.h"
#include "util/hash_util.h"

namespace xllm {

// Scheduler-side prefix cache for Qwen3.5 GDN linear-state checkpoints,
// mirroring the KV-cache architecture (a free-list block manager + an LRU
// prefix cache sharing one id space). All physical slots [0, num_slots) live
// here:
//   - a slot held by a running sequence is "live" (its Block ref_count >= 1);
//   - a slot pinned by a prefix-cache entry is a "checkpoint" (ref_count == 1,
//     held by this cache), evictable via LRU;
//   - everything else is free.
//
// Slot 0 is reserved as the padding slot (kPaddingLinearStateId) by the
// underlying SingleBlockManager and is never handed out.
//
// Unlike KV blocks, GDN recurrent state is overwritten in place and cannot be
// shared, so a live slot always has exactly one owner and restore/save still
// require a device copy on the worker. This cache only owns the scheduler-side
// bookkeeping: which slots are live, which hold a checkpoint, and the LRU order
// used to reclaim checkpoints under pressure. Eviction is purely local here --
// there is no cross-process notification, because the worker is a thin executor
// that copies between the (src, dst) slots this cache dictates.
class LinearStatePrefixCache final {
 public:
  explicit LinearStatePrefixCache(int32_t num_slots);

  LinearStatePrefixCache(const LinearStatePrefixCache&) = delete;
  LinearStatePrefixCache& operator=(const LinearStatePrefixCache&) = delete;

  // Acquire a live slot for a running sequence. Reclaims the least-recently
  // used checkpoint when no slot is free. Returns an invalid Block only when
  // every slot is already live (no checkpoint left to reclaim).
  Block acquire_live_slot();

  // Match the checkpoint slot for a restore hash, refreshing its LRU position.
  // Returns the slot id, or -1 on miss.
  int32_t match(const XXH3Key& restore_hash);

  // Insert a fresh checkpoint slot for save_hash and return its slot id (the
  // copy destination the worker will write). Returns the existing slot id if
  // save_hash is already cached (no new slot, no copy needed). Returns -1 when
  // no slot can be reclaimed.
  int32_t insert(const XXH3Key& save_hash);

  // Same as insert(), but eviction skips checkpoint slots needed by restore ops
  // already resolved for the current batch.
  int32_t insert(const XXH3Key& save_hash,
                 const std::unordered_set<int32_t>& protected_slots);

  // True if a checkpoint exists for the given hash (no LRU refresh).
  bool contains(const XXH3Key& prefix_hash) const;

  int32_t num_slots() const { return num_slots_; }

 private:
  struct CacheEntry {
    Block block;  // pins the slot so the free list cannot hand it out
    std::list<XXH3Key>::iterator lru_it;
  };

  // Reclaim one slot to the free list, evicting the LRU checkpoint if needed.
  // Returns false only when no slot is free and no checkpoint can be evicted.
  bool ensure_free_slot(
      const std::unordered_set<int32_t>& protected_slots = {});

  // Move an existing checkpoint to the most-recently-used end of the LRU.
  void touch(const XXH3Key& prefix_hash, CacheEntry& entry);

  int32_t num_slots_;
  SingleBlockManager slots_;

  std::unordered_map<XXH3Key,
                     CacheEntry,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      cached_slots_;
  std::list<XXH3Key> lru_;  // front = least recently used
};

}  // namespace xllm
