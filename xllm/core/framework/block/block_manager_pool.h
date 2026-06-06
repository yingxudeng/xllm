/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <queue>
#include <vector>

#include "block_manager.h"
#include "framework/block/kv_cache_manager.h"
#include "framework/block/linear_state_prefix_cache.h"
#include "framework/block/single_block_manager.h"
#include "util/hash_util.h"

namespace xllm {

struct LinearStateCacheOp;

class BlockManagerPool : public KVCacheManager {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(uint32_t, host_num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(uint32_t, single_block_capacity) = 0;
    PROPERTY(bool, enable_linear_state) = false;
    // Total physical linear-state slots [0, N) for the unified slot pool
    // (= num_linear_state_blocks). Only used when enable_linear_state is true.
    PROPERTY(int32_t, linear_state_num_slots) = 0;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    PROPERTY(bool, enable_cache_upload) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
    PROPERTY(bool, enable_xtensor) = false;
    PROPERTY(int64_t, num_layers) = 0;  // Required when enable_xtensor is true
    PROPERTY(int64_t, slot_size) = 0;   // Memory size per slot (for xtensor)
    PROPERTY(std::string, model_id);    // Model ID for multi-model support
  };

  explicit BlockManagerPool(const Options& options, int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  virtual bool allocate(Sequence* sequence) override;
  virtual bool allocate(std::vector<Sequence*>& sequences) override;
  virtual bool allocate(Sequence* sequence, size_t num_tokens) override;
  virtual bool allocate(Sequence* sequence,
                        size_t num_tokens,
                        size_t needed_copy_in_blocks_num) override;

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  virtual std::vector<Block> allocate(size_t num_tokens,
                                      int32_t& dp_rank) override;

  virtual bool try_allocate(Sequence* sequence) override;

  virtual void deallocate(Request* request) override;
  virtual void deallocate(std::vector<Sequence*>& sequences) override;
  virtual void deallocate(Sequence* sequence) override;

  void deallocate_without_cache(Sequence* sequence);

  virtual void allocate_shared(Sequence* sequence) override;
  virtual void cache(Sequence* sequence) override;

  virtual std::vector<std::vector<BlockTransferInfo>>*
  get_swap_block_transfer_infos() override;
  virtual void reset_transfer_infos() override;

  virtual void get_merged_kvcache_event(KvCacheEvent* event) const;
  virtual float get_gpu_cache_usage_perc() const;

  // Linear-state prefix cache for the given dp rank, or nullptr when linear
  // state is disabled.
  LinearStatePrefixCache* linear_state_prefix_cache(int32_t dp_rank);

  class LinearStateCheckpointReservations final {
   public:
    LinearStateCheckpointReservations() = default;
    LinearStateCheckpointReservations(
        const LinearStateCheckpointReservations&) = delete;
    LinearStateCheckpointReservations& operator=(
        const LinearStateCheckpointReservations&) = delete;
    LinearStateCheckpointReservations(
        LinearStateCheckpointReservations&&) noexcept = default;
    LinearStateCheckpointReservations& operator=(
        LinearStateCheckpointReservations&&) noexcept = default;

    class Promotion final {
     public:
      Promotion(const XXH3Key& hash,
                Sequence* sequence,
                int32_t live_slot_id,
                Block replacement_slot);

      Promotion(const Promotion&) = delete;
      Promotion& operator=(const Promotion&) = delete;
      Promotion(Promotion&&) noexcept = default;
      Promotion& operator=(Promotion&&) noexcept = default;

      const XXH3Key& hash() const { return hash_; }
      Sequence* sequence() const { return sequence_; }
      int32_t live_slot_id() const { return live_slot_id_; }
      Block take_replacement_slot() { return std::move(replacement_slot_); }

     private:
      XXH3Key hash_;
      Sequence* sequence_ = nullptr;
      int32_t live_slot_id_ = -1;
      Block replacement_slot_;
    };

    int32_t dp_rank = -1;
    // Pins checkpoint slots matched for restore until the worker has copied
    // them into sequence-owned live slots.
    std::vector<Block> restore_pins;
    std::vector<LinearStateCheckpointReservation> reservations;
    std::vector<Promotion> promotions;
  };

  // Resolve linear-state checkpoint copy plans through the scheduler-side
  // cache. Restores are matched before saves reserve slots, so save-side
  // eviction cannot reclaim checkpoints needed by this batch's restores. The
  // returned reservations must be cached only after worker save completes.
  LinearStateCheckpointReservations resolve_linear_state_cache_ops(
      int32_t dp_rank,
      std::vector<LinearStateCacheOp>* cache_ops,
      const std::vector<Sequence*>& sequences = {});

  void cache(LinearStateCheckpointReservations&& reservations);

  virtual uint32_t num_blocks() const override;
  virtual int32_t block_size() const override;
  virtual std::vector<size_t> num_blocks_in_prefix_cache() const override;
  virtual std::vector<size_t> num_free_blocks() const override;
  virtual std::vector<size_t> num_used_blocks() const override;
  virtual double kv_cache_utilization() const override;

  // get the options for the block manager
  const Options& options() const { return options_; }

  // Reserve XTensor padding blocks for each DP manager.
  // Should be called after KV tensors are created.
  void reserve_xtensor_padding_blocks() override;

 protected:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  bool process_beam_search(Sequence* sequence, bool need_swap = false);
  bool allocate_single_block(Sequence* sequence, int32_t dp_rank);
  void deallocate_single_block(Sequence* sequence, int32_t dp_rank);

 protected:
  // the options for the block manager.
  Options options_;

 private:
  void trim_shared_blocks_to_linear_state(int32_t dp_rank,
                                          size_t existed_shared_blocks_num,
                                          size_t max_reusable_blocks,
                                          std::vector<Block>* shared_blocks);

  // Acquire/release the per-sequence linear-state live slot alongside the
  // single block. No-ops when linear state is disabled.
  bool allocate_linear_state_slot(Sequence* sequence, int32_t dp_rank);
  void release_linear_state_slot(Sequence* sequence, int32_t dp_rank);

  std::vector<std::vector<BlockTransferInfo>> swap_block_transfer_infos_;
  std::vector<std::unique_ptr<SingleBlockManager>> single_block_managers_;
  // Linear-state prefix caches, one per dp rank. Empty when linear state is
  // disabled.
  std::vector<std::unique_ptr<LinearStatePrefixCache>>
      linear_state_prefix_caches_;

 protected:
  std::vector<std::unique_ptr<BlockManager>> block_managers_;
};

}  // namespace xllm
