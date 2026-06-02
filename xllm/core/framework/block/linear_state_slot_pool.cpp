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

#include "framework/block/linear_state_slot_pool.h"

#include <glog/logging.h>

#include <iterator>
#include <utility>

namespace xllm {

LinearStateSlotPool::LinearStateSlotPool(int32_t num_slots)
    : num_slots_(num_slots),
      slots_(static_cast<uint32_t>(num_slots),
             /*resource_name=*/"linear state slot",
             /*exhaustion_message=*/"") {
  CHECK_GT(num_slots, 0) << "linear state pool needs at least one slot";
}

bool LinearStateSlotPool::ensure_free_slot() {
  if (slots_.num_free_blocks() > 0) {
    return true;
  }
  if (lru_.empty()) {
    // Every slot is live; nothing can be reclaimed.
    return false;
  }
  const XXH3Key victim = lru_.front();
  lru_.pop_front();
  auto it = checkpoints_.find(victim);
  CHECK(it != checkpoints_.end()) << "LRU and checkpoint map out of sync";
  // Erasing the entry drops its Block handle, which returns the slot to the
  // underlying free list.
  checkpoints_.erase(it);
  return slots_.num_free_blocks() > 0;
}

Block LinearStateSlotPool::acquire_live() {
  if (!ensure_free_slot()) {
    return Block();
  }
  auto blocks = slots_.allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear state slot allocation failed";
  return std::move(blocks[0]);
}

int32_t LinearStateSlotPool::lookup(const XXH3Key& restore_hash) {
  auto it = checkpoints_.find(restore_hash);
  if (it == checkpoints_.end()) {
    return -1;
  }
  touch(restore_hash, it->second);
  return it->second.block.id();
}

int32_t LinearStateSlotPool::checkpoint(const XXH3Key& save_hash) {
  auto it = checkpoints_.find(save_hash);
  if (it != checkpoints_.end()) {
    // Already checkpointed: no new slot and no copy are needed.
    touch(save_hash, it->second);
    return it->second.block.id();
  }
  if (!ensure_free_slot()) {
    return -1;
  }
  auto blocks = slots_.allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear state checkpoint allocation failed";
  const int32_t slot_id = blocks[0].id();
  lru_.push_back(save_hash);
  checkpoints_.emplace(
      save_hash, CheckpointEntry{std::move(blocks[0]), std::prev(lru_.end())});
  return slot_id;
}

bool LinearStateSlotPool::has_checkpoint(const XXH3Key& prefix_hash) const {
  return checkpoints_.find(prefix_hash) != checkpoints_.end();
}

void LinearStateSlotPool::touch(const XXH3Key& prefix_hash,
                                CheckpointEntry& entry) {
  lru_.erase(entry.lru_it);
  lru_.push_back(prefix_hash);
  entry.lru_it = std::prev(lru_.end());
}

}  // namespace xllm
