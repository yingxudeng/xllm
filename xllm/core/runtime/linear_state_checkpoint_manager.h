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

#include <mutex>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

// Worker-side executor for Qwen3.5 GDN linear-state checkpoints. All slot
// management (which slot is live, which holds a checkpoint, and LRU eviction)
// lives in the scheduler's LinearStateSlotPool. This class only performs the
// device copies the scheduler dictates via each LinearStateCacheOp:
//   - restore: copy restore_src_slot_id -> linear_state_id before compute
//   - save:    copy linear_state_id -> save_dst_slot_id after compute
//
// It keeps one piece of state: slot_tags_[slot] records the prefix hash whose
// recurrent state currently lives in that physical slot. Because the scheduler
// reserves save destinations optimistically (a step's inputs may be built
// before the previous step's save has physically run, under schedule overlap),
// restore verifies slot_tags_[restore_src_slot_id] == restore_prefix_hash and
// degrades to a safe cold start when it does not match -- recompute rather than
// copy stale data.
class LinearStateCheckpointManager final {
 public:
  enum class RestoreAction : int8_t {
    // No restore requested for this op (continued request or no prefix reused).
    SKIPPED,
    // Checkpoint copied into the live slot; the slot now holds valid state.
    RESTORED,
    // Restore was requested but the source checkpoint was not physically
    // available; the live slot must be recomputed from scratch.
    COLD_START,
  };

  LinearStateCheckpointManager(std::vector<KVCache>& kv_caches,
                               c10::DeviceIndex device_index);

  LinearStateCheckpointManager(const LinearStateCheckpointManager&) = delete;
  LinearStateCheckpointManager& operator=(const LinearStateCheckpointManager&) =
      delete;

  void initialize();

  // Copy each op's live slot into its reserved save destination (after
  // compute).
  void save(const std::vector<LinearStateCacheOp>& cache_ops);

  // Copy each op's restore source into its live slot (before compute), with a
  // self-check that degrades to COLD_START on a stale/unready source.
  std::vector<RestoreAction> restore(
      const std::vector<LinearStateCacheOp>& cache_ops);

 private:
  // Copy conv + ssm recurrent state between two physical slots across all
  // linear-attention layers. Returns true if at least one layer was copied.
  bool copy_slot(int32_t dst_slot_id, int32_t src_slot_id);

  bool slot_in_range(int32_t slot_id) const {
    return slot_id >= 0 && slot_id < num_slots_;
  }

  std::vector<KVCache>& kv_caches_;
  c10::DeviceIndex device_index_;

  std::mutex mutex_;
  // Prefix hash whose state currently lives in each physical slot; all-zero
  // means unknown/garbage. Sized to the number of physical linear-state slots.
  std::vector<LinearStatePrefixHash> slot_tags_;
  int32_t num_slots_ = 0;
};

}  // namespace xllm
