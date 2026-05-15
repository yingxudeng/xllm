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

#include <torch/torch.h>

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

class LinearStateSnapshotManager final {
 public:
  struct Snapshot {
    int32_t checkpoint_slot_id = -1;
    int32_t ref_count = 0;
    bool pending_delete = false;
  };

  struct SaveUpdate {
    std::vector<LinearStatePrefixHash> saved_prefix_hashes;
    std::vector<LinearStatePrefixHash> evicted_prefix_hashes;
  };

  LinearStateSnapshotManager(std::vector<KVCache>& kv_caches,
                             c10::DeviceIndex device_index,
                             int32_t max_seqs_per_batch);

  LinearStateSnapshotManager(const LinearStateSnapshotManager&) = delete;
  LinearStateSnapshotManager& operator=(const LinearStateSnapshotManager&) =
      delete;

  void initialize();
  SaveUpdate save(const ModelInputParams& input_params);
  std::vector<LinearStatePrefixHash> restore(ModelInputParams& input_params);
  void prune(const ModelInputParams& input_params);
  void release_refs(const std::vector<LinearStatePrefixHash>& prefix_hashes);

 private:
  class PrefixHashHasher {
   public:
    size_t operator()(const LinearStatePrefixHash& prefix_hash) const {
      size_t hash = 0;
      for (const uint8_t value : prefix_hash) {
        hash = hash * 131 + value;
      }
      return hash;
    }
  };

  using SnapshotMap =
      std::unordered_map<LinearStatePrefixHash, Snapshot, PrefixHashHasher>;

  bool save_one(const LinearStatePrefixHash& prefix_hash,
                int32_t linear_state_id,
                std::vector<LinearStatePrefixHash>* evicted);
  bool restore_one(const LinearStatePrefixHash& prefix_hash,
                   int32_t linear_state_id);
  int32_t acquire_checkpoint_slot(const LinearStatePrefixHash& prefix_hash,
                                  std::vector<LinearStatePrefixHash>* evicted);
  void release_checkpoint_slot(int32_t slot_id);
  int32_t erase_snapshot(SnapshotMap::iterator snapshot_it);
  void touch(const LinearStatePrefixHash& prefix_hash);

  std::vector<KVCache>& kv_caches_;
  c10::DeviceIndex device_index_;
  int32_t max_seqs_per_batch_;

  std::mutex mutex_;
  SnapshotMap snapshots_;
  std::list<LinearStatePrefixHash> lru_;
  std::unordered_map<LinearStatePrefixHash,
                     std::list<LinearStatePrefixHash>::iterator,
                     PrefixHashHasher>
      lru_iters_;
  std::vector<int32_t> free_checkpoint_slots_;
#ifndef NDEBUG
  std::vector<uint8_t> checkpoint_slot_free_;
#endif
  int32_t live_slots_ = 0;
  int32_t checkpoint_slots_ = 0;
  std::unordered_map<int32_t, std::string> active_requests_;
};

}  // namespace xllm
