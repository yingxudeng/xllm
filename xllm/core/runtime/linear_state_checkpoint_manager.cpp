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

#include "runtime/linear_state_checkpoint_manager.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#if defined(USE_NPU)
#include "acl/acl.h"
#include "platform/npu/device_capture_lock.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#endif

#include "common/global_flags.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "util/hash_util.h"

namespace xllm {

namespace {

#if defined(USE_NPU)
void synchronize_current_npu_stream(c10::DeviceIndex device_index,
                                    const char* action) {
  aclrtStream current_stream =
      c10_npu::getCurrentNPUStream(device_index).stream();
  aclError status = aclrtSynchronizeStream(current_stream);
  CHECK_EQ(status, ACL_SUCCESS)
      << "aclrtSynchronizeStream failed after Qwen3.5 linear state checkpoint "
      << action << ", error code: " << status;
}
#endif

}  // namespace

LinearStateCheckpointManager::LinearStateCheckpointManager(
    std::vector<KVCache>& kv_caches,
    c10::DeviceIndex device_index,
    int32_t max_seqs_per_batch)
    : kv_caches_(kv_caches),
      device_index_(device_index),
      max_seqs_per_batch_(max_seqs_per_batch) {}

void LinearStateCheckpointManager::initialize() {
  free_checkpoint_slots_.clear();
  live_slots_ = 0;
  checkpoint_slots_ = 0;
  checkpoints_.clear();
  lru_.clear();
  lru_iters_.clear();
  active_requests_.clear();

  int64_t num_linear_state_blocks = 0;
  for (const auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    CHECK_EQ(conv_cache.size(0), ssm_cache.size(0));
    if (num_linear_state_blocks == 0) {
      num_linear_state_blocks = conv_cache.size(0);
      continue;
    }
    CHECK_EQ(num_linear_state_blocks, conv_cache.size(0));
  }

  if (num_linear_state_blocks == 0) {
    return;
  }

  live_slots_ = static_cast<int32_t>(calculate_linear_state_live_slots(
      num_linear_state_blocks, max_seqs_per_batch_));

  checkpoint_slots_ =
      static_cast<int32_t>(num_linear_state_blocks) - live_slots_;
  free_checkpoint_slots_.reserve(checkpoint_slots_);
  for (int32_t slot_id = static_cast<int32_t>(num_linear_state_blocks) - 1;
       slot_id >= live_slots_;
       --slot_id) {
    free_checkpoint_slots_.emplace_back(slot_id);
  }
  LOG(INFO) << "Qwen3.5 linear state checkpoint slots initialized; live_slots="
            << live_slots_ << ", checkpoint_slots=" << checkpoint_slots_;
}

LinearStateCheckpointManager::SaveResult LinearStateCheckpointManager::save(
    const std::vector<LinearStateCacheOp>& cache_ops) {
  SaveResult result;
  if (!FLAGS_enable_prefix_cache) {
    return result;
  }
  if (cache_ops.empty()) {
    return result;
  }

#if defined(USE_NPU)
  std::optional<std::unique_lock<std::mutex>> capture_lock_guard;
  if (FLAGS_enable_graph) {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_index_);
    capture_lock_guard.emplace(capture_lock);
  }
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  bool saved_any = false;
  for (const LinearStateCacheOp& cache_op : cache_ops) {
    const LinearStatePrefixHash& save_prefix_hash = cache_op.save_prefix_hash;
    if (is_zero_prefix_hash(save_prefix_hash)) {
      continue;
    }
    const int32_t linear_state_id = cache_op.linear_state_id;
    if (checkpoint_to_slot(
            save_prefix_hash, linear_state_id, &result.evicted_prefix_hashes)) {
      saved_any = true;
      VLOG(1) << "Qwen3.5 linear state checkpoint saved; linear_state_id="
              << linear_state_id;
    }
  }
#if defined(USE_NPU)
  if (saved_any) {
    synchronize_current_npu_stream(device_index_, "save");
  }
#endif
  return result;
}

std::vector<LinearStateCheckpointManager::RestoreAction>
LinearStateCheckpointManager::restore(
    const std::vector<LinearStateCacheOp>& cache_ops) {
  std::vector<RestoreAction> actions(cache_ops.size(), RestoreAction::SKIPPED);
  if (cache_ops.empty()) {
    return actions;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  bool restored_any = false;
  for (size_t i = 0; i < cache_ops.size(); ++i) {
    const LinearStateCacheOp& cache_op = cache_ops[i];
    const int32_t linear_state_id = cache_op.linear_state_id;
    if (linear_state_id < 0) {
      continue;
    }

    const std::string& request_id = cache_op.request_id;
    auto active_it = active_requests_.find(linear_state_id);
    if (!request_id.empty() && active_it != active_requests_.end() &&
        active_it->second == request_id) {
      actions[i] = RestoreAction::CONTINUED;
      continue;
    }

    const LinearStatePrefixHash& restore_prefix_hash =
        cache_op.restore_prefix_hash;
    if (is_zero_prefix_hash(restore_prefix_hash)) {
      active_requests_[linear_state_id] = request_id;
      actions[i] = RestoreAction::COLD_START;
      continue;
    }

    if (restore_from_slot(restore_prefix_hash, linear_state_id)) {
      VLOG(1) << "Qwen3.5 linear state checkpoint restored; linear_state_id="
              << linear_state_id;
      active_requests_[linear_state_id] = request_id;
      actions[i] = RestoreAction::RESTORED;
      restored_any = true;
      continue;
    }

    LOG(WARNING) << "Qwen3.5 prefix cache hit lacks linear state checkpoint; "
                 << "falling back to recompute, linear_state_id="
                 << linear_state_id;
    active_requests_.erase(linear_state_id);
    actions[i] = RestoreAction::COLD_START;
  }
#if defined(USE_NPU)
  if (restored_any) {
    synchronize_current_npu_stream(device_index_, "restore");
  }
#endif
  return actions;
}

void LinearStateCheckpointManager::evict(
    const std::vector<LinearStatePrefixHash>& prefix_hashes) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const LinearStatePrefixHash& prefix_hash : prefix_hashes) {
    if (is_zero_prefix_hash(prefix_hash)) {
      continue;
    }
    auto checkpoint_it = checkpoints_.find(prefix_hash);
    if (checkpoint_it == checkpoints_.end()) {
      continue;
    }
    auto lru_it = lru_iters_.find(prefix_hash);
    if (lru_it != lru_iters_.end()) {
      lru_.erase(lru_it->second);
      lru_iters_.erase(lru_it);
    }
    release_checkpoint_slot(erase_checkpoint(checkpoint_it));
  }
}

bool LinearStateCheckpointManager::checkpoint_to_slot(
    const LinearStatePrefixHash& prefix_hash,
    int32_t linear_state_id,
    std::vector<LinearStatePrefixHash>* evicted) {
  if (linear_state_id < 0 || linear_state_id >= live_slots_ ||
      checkpoint_slots_ <= 0) {
    return false;
  }

  const bool had_existing =
      checkpoints_.find(prefix_hash) != checkpoints_.end();
  const int32_t checkpoint_slot_id =
      acquire_checkpoint_slot(prefix_hash, evicted);
  if (checkpoint_slot_id < 0) {
    LOG(WARNING) << "No reusable Qwen3.5 linear state checkpoint slot; "
                 << "falling back to recompute.";
    return false;
  }

  bool copied = false;
  for (const auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    conv_cache.select(0, checkpoint_slot_id)
        .copy_(conv_cache.select(0, linear_state_id));
    ssm_cache.select(0, checkpoint_slot_id)
        .copy_(ssm_cache.select(0, linear_state_id));
    copied = true;
  }

  if (!copied) {
    if (!had_existing) {
      release_checkpoint_slot(checkpoint_slot_id);
    }
    return false;
  }

  checkpoints_[prefix_hash] = checkpoint_slot_id;
  touch(prefix_hash);
  return true;
}

bool LinearStateCheckpointManager::restore_from_slot(
    const LinearStatePrefixHash& prefix_hash,
    int32_t linear_state_id) {
  if (linear_state_id < 0 || linear_state_id >= live_slots_) {
    return false;
  }

  auto checkpoint_it = checkpoints_.find(prefix_hash);
  if (checkpoint_it == checkpoints_.end()) {
    return false;
  }

  const int32_t checkpoint_slot_id = checkpoint_it->second;
  if (checkpoint_slot_id < 0) {
    return false;
  }
  bool copied = false;
  for (auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    conv_cache.select(0, linear_state_id)
        .copy_(conv_cache.select(0, checkpoint_slot_id));
    ssm_cache.select(0, linear_state_id)
        .copy_(ssm_cache.select(0, checkpoint_slot_id));
    copied = true;
  }
  if (!copied) {
    return false;
  }
  touch(prefix_hash);
  return true;
}

void LinearStateCheckpointManager::touch(
    const LinearStatePrefixHash& prefix_hash) {
  auto lru_it = lru_iters_.find(prefix_hash);
  if (lru_it != lru_iters_.end()) {
    lru_.erase(lru_it->second);
  }
  lru_.emplace_back(prefix_hash);
  auto tail_it = lru_.end();
  --tail_it;
  lru_iters_[prefix_hash] = tail_it;
}

void LinearStateCheckpointManager::release_checkpoint_slot(
    int32_t checkpoint_slot_id) {
  if (checkpoint_slot_id < live_slots_) {
    return;
  }
  free_checkpoint_slots_.emplace_back(checkpoint_slot_id);
}

int32_t LinearStateCheckpointManager::erase_checkpoint(
    CheckpointMap::iterator it) {
  int32_t slot_id = it->second;
  checkpoints_.erase(it);
  return slot_id;
}

int32_t LinearStateCheckpointManager::acquire_checkpoint_slot(
    const LinearStatePrefixHash& prefix_hash,
    std::vector<LinearStatePrefixHash>* evicted) {
  auto existing_it = checkpoints_.find(prefix_hash);
  if (existing_it != checkpoints_.end()) {
    touch(prefix_hash);
    return existing_it->second;
  }

  if (free_checkpoint_slots_.empty()) {
    while (!lru_.empty()) {
      const LinearStatePrefixHash evict_hash = lru_.front();
      lru_.pop_front();
      lru_iters_.erase(evict_hash);
      auto evict_it = checkpoints_.find(evict_hash);
      if (evict_it == checkpoints_.end()) {
        continue;
      }
      if (evicted != nullptr) {
        evicted->emplace_back(evict_hash);
      }
      return erase_checkpoint(evict_it);
    }
    return -1;
  }

  const int32_t checkpoint_slot_id = free_checkpoint_slots_.back();
  free_checkpoint_slots_.pop_back();
  return checkpoint_slot_id;
}

}  // namespace xllm
