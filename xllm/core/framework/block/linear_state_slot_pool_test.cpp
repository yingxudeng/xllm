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

#include "linear_state_slot_pool.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "util/hash_util.h"

namespace xllm {
namespace {

// Build a distinct prefix hash keyed by a single byte so tests can reference
// checkpoints by an easy-to-read tag.
XXH3Key make_hash(uint8_t tag) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  bytes.fill(tag);
  return XXH3Key(bytes.data());
}

}  // namespace

// Padding slot 0 is never handed out, and a pool of N physical slots exposes
// N - 1 live slots before it runs dry.
TEST(LinearStateSlotPoolTest, PaddingSlotReservedAndLiveSlotsExhaust) {
  LinearStateSlotPool pool(/*num_slots=*/3);
  EXPECT_EQ(pool.num_slots(), 3);

  Block first = pool.acquire_live();
  Block second = pool.acquire_live();
  ASSERT_TRUE(first.is_valid());
  ASSERT_TRUE(second.is_valid());
  EXPECT_GE(first.id(), 1);
  EXPECT_GE(second.id(), 1);
  EXPECT_NE(first.id(), second.id());

  // No free slot and no checkpoint to reclaim -> invalid handle.
  Block third = pool.acquire_live();
  EXPECT_FALSE(third.is_valid());
}

// A live slot returns to the pool when its handle is dropped.
TEST(LinearStateSlotPoolTest, DroppingLiveHandleReleasesSlot) {
  LinearStateSlotPool pool(/*num_slots=*/2);
  int32_t reused_id = -1;
  {
    Block live = pool.acquire_live();
    ASSERT_TRUE(live.is_valid());
    reused_id = live.id();
    // Only one usable slot; it is now live.
    EXPECT_FALSE(pool.acquire_live().is_valid());
  }
  // Handle dropped: the slot is free again.
  Block live_again = pool.acquire_live();
  ASSERT_TRUE(live_again.is_valid());
  EXPECT_EQ(live_again.id(), reused_id);
}

// checkpoint() pins a slot that lookup()/has_checkpoint() can find, and reusing
// the same hash returns the same slot without consuming another.
TEST(LinearStateSlotPoolTest, CheckpointLookupAndDedup) {
  LinearStateSlotPool pool(/*num_slots=*/4);
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = pool.checkpoint(h1);
  EXPECT_GE(slot, 1);
  EXPECT_TRUE(pool.has_checkpoint(h1));
  EXPECT_EQ(pool.lookup(h1), slot);

  // Re-checkpointing the same hash is idempotent.
  EXPECT_EQ(pool.checkpoint(h1), slot);

  // Unknown hashes miss.
  EXPECT_FALSE(pool.has_checkpoint(make_hash(2)));
  EXPECT_EQ(pool.lookup(make_hash(2)), -1);
}

// When all usable slots hold checkpoints, the next checkpoint reclaims the
// least-recently-used one.
TEST(LinearStateSlotPoolTest, CheckpointEvictsLeastRecentlyUsed) {
  // 3 physical slots -> 2 usable checkpoint slots.
  LinearStateSlotPool pool(/*num_slots=*/3);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  pool.checkpoint(h1);
  pool.checkpoint(h2);
  ASSERT_TRUE(pool.has_checkpoint(h1));
  ASSERT_TRUE(pool.has_checkpoint(h2));

  // Pool full: a third checkpoint evicts the oldest (h1).
  pool.checkpoint(h3);
  EXPECT_FALSE(pool.has_checkpoint(h1));
  EXPECT_TRUE(pool.has_checkpoint(h2));
  EXPECT_TRUE(pool.has_checkpoint(h3));
}

// lookup() refreshes LRU order so the touched checkpoint survives eviction.
TEST(LinearStateSlotPoolTest, LookupRefreshesLruOrder) {
  LinearStateSlotPool pool(/*num_slots=*/3);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  pool.checkpoint(h1);
  pool.checkpoint(h2);
  // Touch h1 so h2 becomes the least-recently-used entry.
  EXPECT_GE(pool.lookup(h1), 1);

  pool.checkpoint(h3);
  EXPECT_TRUE(pool.has_checkpoint(h1));
  EXPECT_FALSE(pool.has_checkpoint(h2));
  EXPECT_TRUE(pool.has_checkpoint(h3));
}

// acquire_live() reclaims a checkpoint slot when no slot is free.
TEST(LinearStateSlotPoolTest, AcquireLiveReclaimsCheckpoint) {
  // 2 physical slots -> 1 usable slot.
  LinearStateSlotPool pool(/*num_slots=*/2);
  const XXH3Key h1 = make_hash(1);

  const int32_t checkpoint_slot = pool.checkpoint(h1);
  ASSERT_GE(checkpoint_slot, 1);
  ASSERT_TRUE(pool.has_checkpoint(h1));

  // The only usable slot holds a checkpoint; acquiring a live slot must reclaim
  // it.
  Block live = pool.acquire_live();
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), checkpoint_slot);
  EXPECT_FALSE(pool.has_checkpoint(h1));
}

// Live slots are never reclaimed for checkpoints.
TEST(LinearStateSlotPoolTest, CheckpointCannotReclaimLiveSlots) {
  LinearStateSlotPool pool(/*num_slots=*/2);
  Block live = pool.acquire_live();
  ASSERT_TRUE(live.is_valid());

  // The only usable slot is live, so no checkpoint slot can be reserved.
  EXPECT_EQ(pool.checkpoint(make_hash(1)), -1);
  EXPECT_FALSE(pool.has_checkpoint(make_hash(1)));
}

}  // namespace xllm
