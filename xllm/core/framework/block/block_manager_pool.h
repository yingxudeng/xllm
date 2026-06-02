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
#include "framework/block/linear_state_slot_pool.h"
#include "framework/block/single_block_manager.h"
#include "util/hash_util.h"

namespace xllm {

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

  // Unified linear-state slot pool for the given dp rank, or nullptr when
  // linear state is disabled. Used by the batch builder to resolve restore/save
  // checkpoint slots while constructing LinearStateCacheOp entries.
  LinearStateSlotPool* linear_state_slot_pool(int32_t dp_rank);

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
                                          std::vector<Block>* shared_blocks);

  // Acquire/release the per-sequence linear-state live slot alongside the
  // single block. No-ops when linear state is disabled.
  bool allocate_linear_state_slot(Sequence* sequence, int32_t dp_rank);
  void release_linear_state_slot(Sequence* sequence, int32_t dp_rank);

  std::vector<std::vector<BlockTransferInfo>> swap_block_transfer_infos_;
  std::vector<std::unique_ptr<SingleBlockManager>> single_block_managers_;
  // Unified reference-counted pool for linear-state slots, one per dp rank.
  // Empty when linear state is disabled.
  std::vector<std::unique_ptr<LinearStateSlotPool>> linear_state_slot_pools_;

 protected:
  std::vector<std::unique_ptr<BlockManager>> block_managers_;
};

}  // namespace xllm
