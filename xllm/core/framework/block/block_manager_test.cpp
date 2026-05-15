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

#include <utility>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "common/global_flags.h"
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
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.has_single_block_id());
  EXPECT_GE(seq.get_single_block_id(), 0);
}

TEST(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.enable_linear_state(true);
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
  // unified scheduler-side single-block pool has 2 ids.
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 0);

  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.enable_linear_state(true);
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
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(3)
      .enable_prefix_cache(false);
  options.enable_linear_state(true);
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
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(1)
      .enable_prefix_cache(false);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = MakeSequence(0, /*prompt_tokens=*/{1});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq1));
  EXPECT_EQ(seq1.dp_rank(), 1);
}

TEST(BlockManagerPoolTest, FailedSingleBlockAllocationDoesNotPinDpRank) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 8);

  BlockManagerPool::Options options;
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .single_block_capacity(1)
      .enable_prefix_cache(false);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = MakeSequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = MakeSequence(2, /*prompt_tokens=*/{3});
  EXPECT_FALSE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), -1);

  pool.deallocate(&seq1);
  ASSERT_TRUE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), 1);
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
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(src.has_single_block_id());

  Sequence clone(src);
  EXPECT_FALSE(clone.has_single_block_id());
  EXPECT_EQ(clone.get_single_block_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(clone.has_single_block_id());
  EXPECT_NE(clone.get_single_block_id(), src.get_single_block_id());
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

TEST(BlockManagerPoolTest, DrainsLinearStateEvictions) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(4).host_num_blocks(0).block_size(1).enable_prefix_cache(
      true);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq = MakeSequence(0, /*prompt_tokens=*/{1, 2});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);

  const XXH3Key hash_0(
      cached_seq.kv_state().kv_blocks()[0].get_immutable_hash_value());
  const XXH3Key hash_1(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.set_linear_state_flags(/*dp_rank=*/0, {hash_1});
  pool.deallocate_without_cache(&cached_seq);

  EXPECT_TRUE(pool.drain_linear_state_evictions().empty());

  Sequence pressure_seq = MakeSequence(1, /*prompt_tokens=*/{3, 4, 5});
  ASSERT_TRUE(pool.allocate(&pressure_seq, pressure_seq.num_tokens()));

  std::vector<PrefixHash> evicted = pool.drain_linear_state_evictions();
  EXPECT_EQ(evicted.size(), 1u);
  EXPECT_EQ(to_xxh3_key(evicted[0]).debug_string(), hash_1.debug_string());
}

TEST(BlockManagerPoolTest, LinearStatePrefixCacheMatchesOnlyCheckpointHashes) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key checkpoint_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.deallocate_without_cache(&cached_seq);

  Sequence miss_seq =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<size_t> used_blocks_before_miss = pool.num_used_blocks();
  ASSERT_TRUE(pool.allocate(&miss_seq, miss_seq.num_tokens()));
  EXPECT_EQ(miss_seq.kv_state().shared_kv_blocks_num(), 0u);
  pool.deallocate_without_cache(&miss_seq);
  EXPECT_EQ(pool.num_used_blocks(), used_blocks_before_miss);

  pool.set_linear_state_flags(/*dp_rank=*/0, {checkpoint_hash});
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));

  Sequence hit_seq =
      MakeSequence(2,
                   /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_kv_blocks_num(), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

TEST(BlockManagerPoolTest, LinearStateCheckpointHashesArePrunedWithEvictions) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(5).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key checkpoint_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.set_linear_state_flags(/*dp_rank=*/0, {checkpoint_hash});
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));
  pool.deallocate_without_cache(&cached_seq);

  Sequence pressure_seq = MakeSequence(
      1,
      /*prompt_tokens=*/{
          9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  ASSERT_TRUE(pool.allocate(&pressure_seq, pressure_seq.num_tokens()));
  pool.deallocate_without_cache(&pressure_seq);

  std::vector<PrefixHash> evicted = pool.drain_linear_state_evictions();
  EXPECT_FALSE(evicted.empty());
  EXPECT_FALSE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));

  Sequence recached_seq =
      MakeSequence(2, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&recached_seq));
  recached_seq.kv_state().set_kv_cache_tokens_num(recached_seq.num_tokens());
  pool.cache(&recached_seq);
  pool.deallocate_without_cache(&recached_seq);

  Sequence miss_after_prune =
      MakeSequence(3, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&miss_after_prune, miss_after_prune.num_tokens()));
  EXPECT_EQ(miss_after_prune.kv_state().shared_kv_blocks_num(), 0u);
}

TEST(BlockManagerPoolTest, LinearStateCheckpointHashesArePrunedByWorkerEvict) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(5).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const XXH3Key checkpoint_hash(
      cached_seq.kv_state().kv_blocks()[1].get_immutable_hash_value());
  pool.set_linear_state_flags(/*dp_rank=*/0, {checkpoint_hash});
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));
  pool.deallocate_without_cache(&cached_seq);

  Sequence hit_before_prune =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&hit_before_prune, hit_before_prune.num_tokens()));
  EXPECT_GT(hit_before_prune.kv_state().shared_kv_blocks_num(), 0u);
  pool.deallocate_without_cache(&hit_before_prune);

  pool.clear_linear_state_flags(/*dp_rank=*/0, {checkpoint_hash});
  EXPECT_FALSE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));

  Sequence miss_after_worker_evict =
      MakeSequence(2, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&miss_after_worker_evict,
                            miss_after_worker_evict.num_tokens()));
  EXPECT_EQ(miss_after_worker_evict.kv_state().shared_kv_blocks_num(), 0u);

  pool.set_linear_state_flags(/*dp_rank=*/0, {checkpoint_hash});
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, checkpoint_hash));
}

TEST(BlockManagerPoolTest, RecordedCheckpointsAreImmediatelyVisible) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(10).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.enable_linear_state(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq_ab =
      MakeSequence(0,
                   /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(pool.allocate(&seq_ab));
  seq_ab.kv_state().set_kv_cache_tokens_num(seq_ab.num_tokens());
  pool.cache(&seq_ab);
  const XXH3Key hash_a(
      seq_ab.kv_state().kv_blocks()[1].get_immutable_hash_value());
  const XXH3Key hash_b(
      seq_ab.kv_state().kv_blocks()[2].get_immutable_hash_value());
  pool.deallocate_without_cache(&seq_ab);

  EXPECT_FALSE(pool.has_linear_state(/*dp_rank=*/0, hash_a));
  EXPECT_FALSE(pool.has_linear_state(/*dp_rank=*/0, hash_b));

  pool.set_linear_state_flags(/*dp_rank=*/0, {hash_a, hash_b});
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, hash_a));
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, hash_b));

  pool.clear_linear_state_flags(/*dp_rank=*/0, {hash_a});
  EXPECT_FALSE(pool.has_linear_state(/*dp_rank=*/0, hash_a));
  EXPECT_TRUE(pool.has_linear_state(/*dp_rank=*/0, hash_b));
}

}  // namespace xllm
