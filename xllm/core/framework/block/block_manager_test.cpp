/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "common/global_flags.h"
#include "framework/block/linear_state_prefix_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/request/incremental_decoder.h"

namespace xllm {

namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

using LinearStatePrefixHash = std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN>;

LinearStatePrefixHash MakePrefixHash(uint8_t tag) {
  LinearStatePrefixHash hash{};
  hash.fill(tag);
  return hash;
}

int32_t insert_linear_state_checkpoint(LinearStatePrefixCache* cache,
                                       const XXH3Key& hash) {
  LinearStateCheckpointReservation reservation;
  const int32_t slot = cache->reserve_checkpoint(hash, &reservation);
  if (reservation.valid()) {
    EXPECT_EQ(cache->insert(std::move(reservation)), slot);
  }
  return slot;
}

int32_t insert_linear_state_checkpoint(LinearStatePrefixCache* cache,
                                       const LinearStatePrefixHash& hash) {
  return insert_linear_state_checkpoint(cache, XXH3Key(hash.data()));
}

Sequence MakeSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = prompt_tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "block_manager_pool_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  prompt_tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

BlockManagerPool::Options make_linear_state_pool_options(
    int32_t linear_state_num_slots) {
  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.single_block_capacity(4)
      .enable_linear_state(true)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

}  // namespace

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto block = manager.allocate();
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(manager.allocate(), "No more blocks available");
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      Block block4 = manager.allocate();
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenEnabled) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.has_single_block_id());
  EXPECT_GT(seq.get_single_block_id(), 0);
}

TEST(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = seq1.get_single_block_id();
  pool.deallocate(&seq1);
  EXPECT_FALSE(seq1.has_single_block_id());

  Sequence seq2 = MakeSequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(seq2.get_single_block_id(), id1);
}

TEST(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 0);

  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  // id 0 is reserved for padding, so capacity 3 exposes 2 usable single-block
  // ids, enough for the two sequences allocated after the rollback.
  options.single_block_capacity(3)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // This sequence needs far more KV blocks than available, forcing KV failure
  // after embedding and linear ids are allocated.
  std::vector<int32_t> huge_prompt(100, 1);
  Sequence fail_seq = MakeSequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(fail_seq.has_single_block_id());

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = MakeSequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(seq1.has_single_block_id());
  EXPECT_TRUE(seq2.has_single_block_id());
}

TEST(BlockManagerPoolTest, SingleBlockCapacityCanBeLowerThanMaxSeqs) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 8);

  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 4 exposes 3 usable single blocks.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(4)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq0 = MakeSequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{2});
  Sequence seq2 = MakeSequence(2, /*prompt_tokens=*/{3});
  Sequence seq3 = MakeSequence(3, /*prompt_tokens=*/{4});

  EXPECT_TRUE(pool.try_allocate(&seq0));
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_FALSE(pool.try_allocate(&seq3));

  EXPECT_TRUE(seq0.has_single_block_id());
  EXPECT_TRUE(seq1.has_single_block_id());
  EXPECT_TRUE(seq2.has_single_block_id());
  EXPECT_FALSE(seq3.has_single_block_id());
}

TEST(BlockManagerPoolTest, DpRankSelectionSkipsExhaustedSingleBlockPool) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 8);

  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(2)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = MakeSequence(0, /*prompt_tokens=*/{1});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq1));
  EXPECT_EQ(seq1.dp_rank(), 1);
}

TEST(BlockManagerPoolTest, SingleBlockExhaustionBehavesLikeKvBlockExhaustion) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 8);

  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(2)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = MakeSequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = MakeSequence(2, /*prompt_tokens=*/{3});
  EXPECT_FALSE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), 0);

  pool.deallocate(&seq0);
  ASSERT_TRUE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), 0);
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.has_single_block_id());
}

TEST(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(src.has_single_block_id());
  ASSERT_TRUE(src.has_linear_state_slot());
  const int32_t src_single_block_id = src.get_single_block_id();
  const int32_t src_linear_state_slot_id = src.get_linear_state_slot_id();
  src.mark_linear_state_initialized();

  Sequence clone(src);
  EXPECT_FALSE(clone.has_single_block_id());
  EXPECT_EQ(clone.get_single_block_id(), -1);
  EXPECT_FALSE(clone.has_linear_state_slot());
  EXPECT_EQ(clone.get_linear_state_slot_id(), -1);
  EXPECT_FALSE(clone.linear_state_initialized());

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(clone.has_single_block_id());
  EXPECT_NE(clone.get_single_block_id(), src_single_block_id);
  EXPECT_TRUE(clone.has_linear_state_slot());
  EXPECT_NE(clone.get_linear_state_slot_id(), src_linear_state_slot_id);
  EXPECT_FALSE(clone.linear_state_initialized());
}

TEST(BlockManagerPoolTest, AllocateAfterPrefixCacheHitAllocatesSuffixBlocks) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  pool.deallocate(&cached_seq);

  Sequence hit_seq =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_kv_blocks_num(), 2);
  EXPECT_GE(hit_seq.kv_state().current_max_tokens_capacity(),
            hit_seq.num_tokens());
}

TEST(BlockManagerPoolTest, LinearStatePrefixCacheMatchesOnlyCheckpointHashes) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key checkpoint_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.deallocate_without_cache(&cached_seq);

  // Without a linear-state checkpoint for the prefix boundary, prefix reuse is
  // trimmed away: the recurrent state cannot be restored, so a hit would be
  // unsafe.
  Sequence miss_seq =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<size_t> used_blocks_before_miss = pool.num_used_blocks();
  ASSERT_TRUE(pool.allocate(&miss_seq, miss_seq.num_tokens()));
  EXPECT_EQ(miss_seq.kv_state().shared_kv_blocks_num(), 0u);
  pool.deallocate_without_cache(&miss_seq);
  EXPECT_EQ(pool.num_used_blocks(), used_blocks_before_miss);

  // Pin a checkpoint for the boundary hash directly in the slot pool, the same
  // way the scheduler does while resolving cache ops. Now the prefix is
  // reusable up to that boundary.
  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash), 1);
  EXPECT_TRUE(prefix_cache->contains(checkpoint_hash));

  Sequence hit_seq =
      MakeSequence(2,
                   /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_kv_blocks_num(), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

// An exact prompt match must leave at least one token for the current forward.
// The matched tail boundary (h2 here) is checkpointed, but reusing it would pop
// back to the previous, uncheckpointed boundary (h1) and lose restorable state,
// so prefix reuse must be 0 instead.
TEST(BlockManagerPoolTest, ExactPromptCannotReuseUncheckpointedTailBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key tail_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.deallocate_without_cache(&cached_seq);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 8-token prompt: max_reusable_blocks = floor((8 - 1) / 4) = 1, so the
  // checkpointed boundary at block 2 is out of reach and reuse falls to 0.
  Sequence exact_seq =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&exact_seq, exact_seq.num_tokens()));
  EXPECT_EQ(exact_seq.kv_state().shared_kv_blocks_num(), 0u);
  pool.deallocate_without_cache(&exact_seq);
}

// One token past the checkpoint boundary leaves work for the current forward,
// so the checkpointed boundary becomes reusable.
TEST(BlockManagerPoolTest, PromptPastCheckpointReusesCheckpointBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key tail_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.deallocate_without_cache(&cached_seq);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // 9-token prompt: max_reusable_blocks = floor((9 - 1) / 4) = 2, so the
  // checkpointed boundary at block 2 is reusable.
  Sequence hit_seq =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_kv_blocks_num(), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

TEST(BlockManagerPoolTest, ResolveLinearStateOpsPinsRestoresFromSaves) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/3),
                        /*dp_size=*/1);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  const LinearStatePrefixHash restore_hash = MakePrefixHash(1);
  const LinearStatePrefixHash evictable_hash = MakePrefixHash(2);
  const LinearStatePrefixHash save_hash = MakePrefixHash(3);
  const int32_t restore_slot =
      insert_linear_state_checkpoint(prefix_cache, restore_hash);
  ASSERT_GE(restore_slot, 1);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, evictable_hash), 1);

  std::vector<LinearStateCacheOp> cache_ops(2);
  cache_ops[0].linear_state_id = 1;
  cache_ops[0].save_prefix_hash = save_hash;
  cache_ops[1].linear_state_id = 2;
  cache_ops[1].restore_prefix_hash = restore_hash;

  auto commit = pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops);

  EXPECT_EQ(commit.dp_rank, 0);
  ASSERT_EQ(commit.restore_pins.size(), 1u);
  EXPECT_EQ(commit.restore_pins[0].id(), restore_slot);
  EXPECT_EQ(cache_ops[1].restore_src_slot_id, restore_slot);
  EXPECT_GE(cache_ops[0].save_dst_slot_id, 1);
  EXPECT_TRUE(prefix_cache->contains(XXH3Key(restore_hash.data())));
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(evictable_hash.data())));
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));

  pool.cache(std::move(commit));
  EXPECT_TRUE(prefix_cache->contains(XXH3Key(save_hash.data())));
}

TEST(BlockManagerPoolTest, ResolveLinearStateOpsSkipsSaveWhenRestoresPinned) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/3),
                        /*dp_size=*/1);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  const LinearStatePrefixHash restore_hash1 = MakePrefixHash(1);
  const LinearStatePrefixHash restore_hash2 = MakePrefixHash(2);
  const LinearStatePrefixHash save_hash = MakePrefixHash(3);
  const int32_t restore_slot1 =
      insert_linear_state_checkpoint(prefix_cache, restore_hash1);
  const int32_t restore_slot2 =
      insert_linear_state_checkpoint(prefix_cache, restore_hash2);
  ASSERT_GE(restore_slot1, 1);
  ASSERT_GE(restore_slot2, 1);

  std::vector<LinearStateCacheOp> cache_ops(3);
  cache_ops[0].linear_state_id = 1;
  cache_ops[0].restore_prefix_hash = restore_hash1;
  cache_ops[1].linear_state_id = 2;
  cache_ops[1].restore_prefix_hash = restore_hash2;
  cache_ops[2].linear_state_id = 3;
  cache_ops[2].save_prefix_hash = save_hash;

  auto commit = pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops);

  EXPECT_EQ(commit.dp_rank, 0);
  ASSERT_EQ(commit.restore_pins.size(), 2u);
  EXPECT_EQ(commit.restore_pins[0].id(), restore_slot1);
  EXPECT_EQ(commit.restore_pins[1].id(), restore_slot2);
  EXPECT_TRUE(commit.reservations.empty());
  EXPECT_EQ(cache_ops[0].restore_src_slot_id, restore_slot1);
  EXPECT_EQ(cache_ops[1].restore_src_slot_id, restore_slot2);
  EXPECT_EQ(cache_ops[2].save_dst_slot_id, -1);
  EXPECT_TRUE(prefix_cache->contains(XXH3Key(restore_hash1.data())));
  EXPECT_TRUE(prefix_cache->contains(XXH3Key(restore_hash2.data())));
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));
}

TEST(BlockManagerPoolTest, ResolveLinearStateOpsPromotesLiveSlot) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  Sequence sequence =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(4);
  sequence.mark_linear_state_initialized();
  const int32_t old_live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(old_live_slot, 1);

  const LinearStatePrefixHash save_hash = MakePrefixHash(9);
  std::vector<LinearStateCacheOp> cache_ops(1);
  cache_ops[0].linear_state_id = old_live_slot;
  cache_ops[0].save_prefix_hash = save_hash;
  std::vector<Sequence*> sequences = {&sequence};

  auto commit =
      pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops, sequences);

  EXPECT_EQ(cache_ops[0].save_dst_slot_id, old_live_slot);
  EXPECT_TRUE(commit.reservations.empty());
  ASSERT_EQ(commit.promotions.size(), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), old_live_slot);
  EXPECT_TRUE(sequence.linear_state_initialized());

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));

  pool.cache(std::move(commit));

  EXPECT_TRUE(prefix_cache->contains(XXH3Key(save_hash.data())));
  Block matched = prefix_cache->match(XXH3Key(save_hash.data()));
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), old_live_slot);
  const int32_t new_live_slot = sequence.get_linear_state_slot_id();
  EXPECT_NE(new_live_slot, old_live_slot);
  EXPECT_FALSE(sequence.linear_state_initialized());

  std::vector<LinearStateCacheOp> restore_ops(1);
  restore_ops[0].linear_state_id = new_live_slot;
  restore_ops[0].restore_prefix_hash = save_hash;
  auto restore_commit =
      pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &restore_ops);
  EXPECT_EQ(restore_ops[0].restore_src_slot_id, old_live_slot);
  EXPECT_TRUE(restore_commit.reservations.empty());
  EXPECT_TRUE(restore_commit.promotions.empty());

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest, ResolveLinearStateOpsDoesNotPromoteDecodeStage) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  Sequence sequence = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_tokens());
  sequence.mark_linear_state_initialized();
  const int32_t live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(live_slot, 1);

  const LinearStatePrefixHash save_hash = MakePrefixHash(8);
  std::vector<LinearStateCacheOp> cache_ops(1);
  cache_ops[0].linear_state_id = live_slot;
  cache_ops[0].save_prefix_hash = save_hash;
  std::vector<Sequence*> sequences = {&sequence};

  auto commit =
      pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops, sequences);

  EXPECT_NE(cache_ops[0].save_dst_slot_id, live_slot);
  EXPECT_TRUE(commit.promotions.empty());
  ASSERT_EQ(commit.reservations.size(), 1u);
  EXPECT_TRUE(sequence.linear_state_initialized());

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));

  pool.cache(std::move(commit));

  EXPECT_TRUE(prefix_cache->contains(XXH3Key(save_hash.data())));
  EXPECT_EQ(sequence.get_linear_state_slot_id(), live_slot);
  EXPECT_TRUE(sequence.linear_state_initialized());

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest, PromotionCommitSkipsWhenLiveSlotChanged) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  Sequence sequence =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(4);
  sequence.mark_linear_state_initialized();
  const int32_t old_live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(old_live_slot, 1);

  const LinearStatePrefixHash save_hash = MakePrefixHash(10);
  std::vector<LinearStateCacheOp> cache_ops(1);
  cache_ops[0].linear_state_id = old_live_slot;
  cache_ops[0].save_prefix_hash = save_hash;
  std::vector<Sequence*> sequences = {&sequence};

  auto commit =
      pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops, sequences);

  EXPECT_EQ(cache_ops[0].save_dst_slot_id, old_live_slot);
  ASSERT_EQ(commit.promotions.size(), 1u);
  sequence.reset_linear_state_slot();

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  pool.cache(std::move(commit));

  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));
  EXPECT_FALSE(sequence.has_linear_state_slot());

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest, ResolveLinearStateOpsCoalescesDuplicateSaves) {
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/3),
                        /*dp_size=*/1);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  const LinearStatePrefixHash save_hash = MakePrefixHash(7);
  std::vector<LinearStateCacheOp> cache_ops(2);
  cache_ops[0].linear_state_id = 1;
  cache_ops[0].save_prefix_hash = save_hash;
  cache_ops[1].linear_state_id = 2;
  cache_ops[1].save_prefix_hash = save_hash;

  auto commit = pool.resolve_linear_state_cache_ops(/*dp_rank=*/0, &cache_ops);

  ASSERT_GE(cache_ops[0].save_dst_slot_id, 1);
  EXPECT_EQ(cache_ops[1].save_dst_slot_id, cache_ops[0].save_dst_slot_id);
  EXPECT_EQ(commit.reservations.size(), 1u);
  EXPECT_FALSE(prefix_cache->contains(XXH3Key(save_hash.data())));

  pool.cache(std::move(commit));
  EXPECT_TRUE(prefix_cache->contains(XXH3Key(save_hash.data())));
  Block matched = prefix_cache->match(XXH3Key(save_hash.data()));
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), cache_ops[0].save_dst_slot_id);
}

TEST(BlockManagerPoolTest, SparseLinearStateCheckpointTrimsSharedKVBlocks) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  const auto check_shared_blocks = [](int32_t checkpoint_block_index,
                                      size_t expected_shared_blocks) {
    BlockManagerPool::Options options;
    options.num_blocks(48).host_num_blocks(0).block_size(4).enable_prefix_cache(
        true);
    options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
        .enable_linear_state(true)
        .linear_state_num_slots(64);
    BlockManagerPool pool(options, /*dp_size=*/1);

    Sequence cached_seq = MakeSequence(
        0,
        /*prompt_tokens=*/{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    ASSERT_TRUE(pool.allocate(&cached_seq));
    cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
    pool.cache(&cached_seq);

    LinearStatePrefixCache* prefix_cache =
        pool.linear_state_prefix_cache(/*dp_rank=*/0);
    ASSERT_NE(prefix_cache, nullptr);
    if (checkpoint_block_index > 0) {
      const size_t block_idx = static_cast<size_t>(checkpoint_block_index - 1);
      const XXH3Key checkpoint_hash(cached_seq.kv_state()
                                        .kv_blocks()[block_idx]
                                        .get_immutable_hash_value());
      EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash),
                1);
    }
    pool.deallocate_without_cache(&cached_seq);

    Sequence hit_seq = MakeSequence(
        1,
        /*prompt_tokens=*/{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
    ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
    EXPECT_EQ(hit_seq.kv_state().shared_kv_blocks_num(),
              expected_shared_blocks);
    pool.deallocate_without_cache(&hit_seq);
  };

  // KV prefix cache stores h1-h4. The linear-state cache may store only one of
  // those shared hashes; reuse must stop at the latest checkpoint-backed hash.
  check_shared_blocks(/*checkpoint_block_index=*/4,
                      /*expected_shared_blocks=*/4);
  check_shared_blocks(/*checkpoint_block_index=*/3,
                      /*expected_shared_blocks=*/3);
  check_shared_blocks(/*checkpoint_block_index=*/0,
                      /*expected_shared_blocks=*/0);
}

TEST(BlockManagerPoolTest, ExactPromptStopsAtEarlierCheckpoint) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.single_block_capacity(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0,
                   /*prompt_tokens=*/{
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key inner_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  const XXH3Key tail_hash(
      cached_seq.kv_state().kv_blocks()[3].get_immutable_hash_value());
  pool.deallocate_without_cache(&cached_seq);

  LinearStatePrefixCache* prefix_cache =
      pool.linear_state_prefix_cache(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, inner_hash), 1);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 16-token prompt: max_reusable_blocks = floor((16 - 1) / 4) = 3, so
  // the tail checkpoint at block 4 is out of reach; reuse stops at the inner
  // checkpoint (block 2).
  Sequence exact_seq =
      MakeSequence(1,
                   /*prompt_tokens=*/{
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&exact_seq, exact_seq.num_tokens()));
  EXPECT_EQ(exact_seq.kv_state().shared_kv_blocks_num(), 2u);
  pool.deallocate_without_cache(&exact_seq);
}

}  // namespace xllm
