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

#include <torch/torch.h>

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

class LinearStateCheckpointManager final {
 public:
  enum class RestoreAction : int8_t {
    SKIPPED,
    CONTINUED,
    RESTORED,
    COLD_START,
  };

  struct SaveResult {
    std::vector<LinearStatePrefixHash> evicted_prefix_hashes;
  };

  LinearStateCheckpointManager(std::vector<KVCache>& kv_caches,
                               c10::DeviceIndex device_index,
                               int32_t max_seqs_per_batch);

  LinearStateCheckpointManager(const LinearStateCheckpointManager&) = delete;
  LinearStateCheckpointManager& operator=(const LinearStateCheckpointManager&) =
      delete;

  void initialize();
  SaveResult save(const std::vector<LinearStateCacheOp>& cache_ops);
  std::vector<RestoreAction> restore(
      const std::vector<LinearStateCacheOp>& cache_ops);
  void evict(const std::vector<LinearStatePrefixHash>& prefix_hashes);

 private:
  using CheckpointMap =
      std::unordered_map<LinearStatePrefixHash, int32_t, PrefixHashHash>;

  bool checkpoint_to_slot(const LinearStatePrefixHash& prefix_hash,
                          int32_t linear_state_id,
                          std::vector<LinearStatePrefixHash>* evicted);
  bool restore_from_slot(const LinearStatePrefixHash& prefix_hash,
                         int32_t linear_state_id);
  // Copies conv/ssm checkpoint state from src_slot_id into dst_slot_id across
  // all linear-attention layers. Returns false if no cache was copied.
  bool copy_checkpoint_slots(int32_t dst_slot_id, int32_t src_slot_id);
  int32_t acquire_checkpoint_slot(const LinearStatePrefixHash& prefix_hash,
                                  std::vector<LinearStatePrefixHash>* evicted);
  void release_checkpoint_slot(int32_t slot_id);
  int32_t erase_checkpoint(CheckpointMap::iterator it);
  void touch(const LinearStatePrefixHash& prefix_hash);

  std::vector<KVCache>& kv_caches_;
  c10::DeviceIndex device_index_;
  int32_t max_seqs_per_batch_;

  std::mutex mutex_;
  CheckpointMap checkpoints_;
  std::list<LinearStatePrefixHash> lru_;
  std::unordered_map<LinearStatePrefixHash,
                     std::list<LinearStatePrefixHash>::iterator,
                     PrefixHashHash>
      lru_iters_;
  std::vector<int32_t> free_checkpoint_slots_;
  std::vector<LinearStatePrefixHash> pending_evicted_prefix_hashes_;
  int32_t live_slots_ = 0;
  int32_t checkpoint_slots_ = 0;
  std::unordered_map<int32_t, std::string> active_requests_;
};

}  // namespace xllm
