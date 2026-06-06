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

#include "framework/block/block.h"
#include "framework/block/single_block_manager.h"
#include "util/hash_util.h"

namespace xllm {

class LinearStateCheckpointReservation final {
 public:
  LinearStateCheckpointReservation();

  LinearStateCheckpointReservation(const LinearStateCheckpointReservation&) =
      delete;
  LinearStateCheckpointReservation& operator=(
      const LinearStateCheckpointReservation&) = delete;

  LinearStateCheckpointReservation(
      LinearStateCheckpointReservation&&) noexcept = default;
  LinearStateCheckpointReservation& operator=(
      LinearStateCheckpointReservation&&) noexcept = default;

  bool valid() const { return block_.is_valid(); }

 private:
  friend class LinearStatePrefixCache;

  LinearStateCheckpointReservation(const XXH3Key& hash, Block block);
  void reset();

  XXH3Key hash_;
  Block block_;
};

// Scheduler-side prefix cache for Qwen3.5 GDN linear-state checkpoints,
// mirroring the KV-cache architecture (a free-list block manager + an LRU
// prefix cache sharing one id space). All physical slots [0, num_slots) live
// here:
//   - a slot held by a running sequence is "live" (its Block ref_count >= 1);
//   - a slot held by a reservation is a pending checkpoint and is invisible to
//     prefix-cache matching;
//   - a slot pinned by a prefix-cache entry is a committed checkpoint
//     (ref_count == 1, held by this cache), evictable via LRU;
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
  // every slot is already live or pinned (no checkpoint left to reclaim).
  Block acquire_live_slot();

  // Match the checkpoint slot for a prefix hash, refreshing its LRU position.
  // The returned Block pins the checkpoint slot until the caller drops it.
  Block match(const XXH3Key& prefix_hash);

  // Reserve a checkpoint slot that the worker will fill later. The returned
  // slot is pinned by `reservation` and remains invisible to match()/contains()
  // until insert() moves it into the committed prefix cache.
  int32_t reserve_checkpoint(const XXH3Key& save_hash,
                             LinearStateCheckpointReservation* reservation);

  // Insert a reserved checkpoint after the worker has copied recurrent state
  // into the reserved slot. If the hash was already cached by another path,
  // the existing cached entry wins and the reservation is released.
  int32_t insert(LinearStateCheckpointReservation&& reservation);

  // Insert a checkpoint that already contains saved recurrent state. Used when
  // a sequence-owned live slot is promoted directly into the committed cache.
  int32_t insert_checkpoint(const XXH3Key& prefix_hash, Block checkpoint_block);

  // True if a checkpoint exists for the given hash (no LRU refresh).
  bool contains(const XXH3Key& prefix_hash) const;

 private:
  struct CacheEntry {
    Block block;  // pins the slot so the free list cannot hand it out
    std::list<XXH3Key>::iterator lru_it;
  };

  // Reclaim one slot to the free list, evicting the LRU checkpoint if needed.
  // Returns false only when no slot is free and every checkpoint is pinned.
  bool ensure_free_slot();

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
