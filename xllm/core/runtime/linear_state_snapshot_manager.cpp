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

#include "linear_state_snapshot_manager.h"

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
      << "aclrtSynchronizeStream failed after Qwen3.5 linear state snapshot "
      << action << ", error code: " << status;
}
#endif

}  // namespace

LinearStateSnapshotManager::LinearStateSnapshotManager(
    std::vector<KVCache>& kv_caches,
    c10::DeviceIndex device_index,
    int32_t max_seqs_per_batch)
    : kv_caches_(kv_caches),
      device_index_(device_index),
      max_seqs_per_batch_(max_seqs_per_batch) {}

void LinearStateSnapshotManager::initialize() {
  free_checkpoint_slots_.clear();
#ifndef NDEBUG
  checkpoint_slot_free_.clear();
#endif
  live_slots_ = 0;
  checkpoint_slots_ = 0;
  snapshots_.clear();
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
  CHECK_GT(live_slots_, 0);
  CHECK_LE(live_slots_, num_linear_state_blocks);

  checkpoint_slots_ =
      static_cast<int32_t>(num_linear_state_blocks) - live_slots_;
  free_checkpoint_slots_.reserve(checkpoint_slots_);
#ifndef NDEBUG
  checkpoint_slot_free_.assign(checkpoint_slots_, 1);
#endif
  for (int32_t slot_id = static_cast<int32_t>(num_linear_state_blocks) - 1;
       slot_id >= live_slots_;
       --slot_id) {
    free_checkpoint_slots_.emplace_back(slot_id);
  }
  LOG(INFO) << "Qwen3.5 linear state cache slots initialized; live_slots="
            << live_slots_ << ", checkpoint_slots=" << checkpoint_slots_;
}

LinearStateSnapshotManager::SaveUpdate LinearStateSnapshotManager::save(
    const ModelInputParams& input_params) {
  SaveUpdate update;
  const std::vector<LinearStateCacheOp>& cache_ops =
      input_params.linear_state_cache_ops;
  if (!FLAGS_enable_prefix_cache) {
    return update;
  }
  if (cache_ops.empty()) {
    return update;
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
  for (const LinearStateCacheOp& cache_op : cache_ops) {
    const LinearStatePrefixHash& save_prefix_hash = cache_op.save_prefix_hash;
    if (is_zero_prefix_hash(save_prefix_hash)) {
      continue;
    }
    const int32_t linear_state_id = cache_op.linear_state_id;
    if (save_one(
            save_prefix_hash, linear_state_id, &update.evicted_prefix_hashes)) {
      update.saved_prefix_hashes.emplace_back(save_prefix_hash);
      VLOG(1) << "Qwen3.5 linear state snapshot saved; linear_state_id="
              << linear_state_id;
    }
  }
  return update;
}

std::vector<LinearStatePrefixHash> LinearStateSnapshotManager::restore(
    ModelInputParams& input_params) {
  std::vector<LinearStatePrefixHash> restored_prefix_hashes;
  const std::vector<LinearStateCacheOp>& cache_ops =
      input_params.linear_state_cache_ops;
  if (cache_ops.empty()) {
    return restored_prefix_hashes;
  }

  CHECK_EQ(input_params.has_initial_state.size(), cache_ops.size())
      << "has_initial_state must be initialized before linear state restore.";
  restored_prefix_hashes.reserve(cache_ops.size());
  std::lock_guard<std::mutex> lock(mutex_);
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
      input_params.has_initial_state[i] = 1;
      continue;
    }

    const LinearStatePrefixHash& restore_prefix_hash =
        cache_op.restore_prefix_hash;
    if (is_zero_prefix_hash(restore_prefix_hash)) {
      active_requests_[linear_state_id] = request_id;
      continue;
    }

    if (restore_one(restore_prefix_hash, linear_state_id)) {
      VLOG(1) << "Qwen3.5 linear state snapshot restored; linear_state_id="
              << linear_state_id;
      input_params.has_initial_state[i] = 1;
      active_requests_[linear_state_id] = request_id;
      restored_prefix_hashes.emplace_back(restore_prefix_hash);
      continue;
    }

    LOG(WARNING) << "Qwen3.5 prefix cache hit lacks linear state snapshot; "
                 << "falling back to recompute, linear_state_id="
                 << linear_state_id;
    active_requests_.erase(linear_state_id);
  }
#if defined(USE_NPU)
  if (!restored_prefix_hashes.empty()) {
    synchronize_current_npu_stream(device_index_, "restore");
  }
#endif
  return restored_prefix_hashes;
}

void LinearStateSnapshotManager::prune(const ModelInputParams& input_params) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const LinearStatePrefixHash& prefix_hash :
       input_params.linear_state_evict_prefix_hashes) {
    if (is_zero_prefix_hash(prefix_hash)) {
      continue;
    }
    auto snapshot_it = snapshots_.find(prefix_hash);
    if (snapshot_it == snapshots_.end()) {
      continue;
    }
    auto lru_it = lru_iters_.find(prefix_hash);
    if (lru_it != lru_iters_.end()) {
      lru_.erase(lru_it->second);
      lru_iters_.erase(lru_it);
    }
    if (snapshot_it->second.ref_count > 0) {
      snapshot_it->second.pending_delete = true;
      continue;
    }
    release_checkpoint_slot(erase_snapshot(snapshot_it));
  }
}

void LinearStateSnapshotManager::release_refs(
    const std::vector<LinearStatePrefixHash>& prefix_hashes) {
  if (prefix_hashes.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (const LinearStatePrefixHash& prefix_hash : prefix_hashes) {
    auto snapshot_it = snapshots_.find(prefix_hash);
    if (snapshot_it == snapshots_.end()) {
      continue;
    }
    CHECK_GT(snapshot_it->second.ref_count, 0);
    --snapshot_it->second.ref_count;
    if (snapshot_it->second.ref_count == 0 &&
        snapshot_it->second.pending_delete) {
      release_checkpoint_slot(erase_snapshot(snapshot_it));
    }
  }
}

bool LinearStateSnapshotManager::save_one(
    const LinearStatePrefixHash& prefix_hash,
    int32_t linear_state_id,
    std::vector<LinearStatePrefixHash>* evicted_prefix_hashes) {
  if (linear_state_id < 0 || linear_state_id >= live_slots_ ||
      checkpoint_slots_ <= 0) {
    return false;
  }

  const bool had_existing_snapshot =
      snapshots_.find(prefix_hash) != snapshots_.end();
  const int32_t checkpoint_slot_id =
      acquire_checkpoint_slot(prefix_hash, evicted_prefix_hashes);
  if (checkpoint_slot_id < 0) {
    LOG(WARNING) << "No reusable Qwen3.5 linear state checkpoint slot; "
                 << "falling back to recompute.";
    return false;
  }

  bool copied_snapshot = false;
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
    copied_snapshot = true;
  }
#if defined(USE_NPU)
  if (copied_snapshot) {
    synchronize_current_npu_stream(device_index_, "save");
  }
#endif

  if (!copied_snapshot) {
    if (!had_existing_snapshot) {
      release_checkpoint_slot(checkpoint_slot_id);
    }
    return false;
  }

  Snapshot snapshot;
  snapshot.checkpoint_slot_id = checkpoint_slot_id;
  snapshots_[prefix_hash] = snapshot;
  touch(prefix_hash);
  return true;
}

bool LinearStateSnapshotManager::restore_one(
    const LinearStatePrefixHash& prefix_hash,
    int32_t linear_state_id) {
  if (linear_state_id < 0 || linear_state_id >= live_slots_) {
    return false;
  }

  auto snapshot_it = snapshots_.find(prefix_hash);
  if (snapshot_it == snapshots_.end()) {
    return false;
  }

  const auto& snapshot = snapshot_it->second;
  if (snapshot.pending_delete || snapshot.checkpoint_slot_id < 0) {
    return false;
  }
  bool copied_snapshot = false;
  for (auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    conv_cache.select(0, linear_state_id)
        .copy_(conv_cache.select(0, snapshot.checkpoint_slot_id));
    ssm_cache.select(0, linear_state_id)
        .copy_(ssm_cache.select(0, snapshot.checkpoint_slot_id));
    copied_snapshot = true;
  }
  if (!copied_snapshot) {
    return false;
  }
  ++snapshot_it->second.ref_count;
  touch(prefix_hash);
  return true;
}

void LinearStateSnapshotManager::touch(
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

void LinearStateSnapshotManager::release_checkpoint_slot(
    int32_t checkpoint_slot_id) {
  DCHECK_GE(checkpoint_slot_id, live_slots_)
      << "release_checkpoint_slot called with live slot: "
      << checkpoint_slot_id;
  if (checkpoint_slot_id < live_slots_) {
    return;
  }
  CHECK_LT(checkpoint_slot_id, live_slots_ + checkpoint_slots_);
#ifndef NDEBUG
  const size_t free_state_index = checkpoint_slot_id - live_slots_;
  DCHECK_LT(free_state_index, checkpoint_slot_free_.size());
  DCHECK_EQ(checkpoint_slot_free_[free_state_index], 0)
      << "duplicate Qwen3.5 linear state checkpoint slot release: "
      << checkpoint_slot_id;
  checkpoint_slot_free_[free_state_index] = 1;
#endif
  free_checkpoint_slots_.emplace_back(checkpoint_slot_id);
}

int32_t LinearStateSnapshotManager::erase_snapshot(
    SnapshotMap::iterator snapshot_it) {
  int32_t slot_id = snapshot_it->second.checkpoint_slot_id;
  snapshots_.erase(snapshot_it);
  return slot_id;
}

int32_t LinearStateSnapshotManager::acquire_checkpoint_slot(
    const LinearStatePrefixHash& prefix_hash,
    std::vector<LinearStatePrefixHash>* evicted_prefix_hashes) {
  auto existing_it = snapshots_.find(prefix_hash);
  if (existing_it != snapshots_.end()) {
    CHECK(!existing_it->second.pending_delete);
    touch(prefix_hash);
    return existing_it->second.checkpoint_slot_id;
  }

  if (free_checkpoint_slots_.empty()) {
    while (!lru_.empty()) {
      const LinearStatePrefixHash evict_hash = lru_.front();
      lru_.pop_front();
      lru_iters_.erase(evict_hash);
      auto evict_it = snapshots_.find(evict_hash);
      if (evict_it == snapshots_.end()) {
        continue;
      }
      if (evicted_prefix_hashes != nullptr) {
        evicted_prefix_hashes->emplace_back(evict_hash);
      }
      if (evict_it->second.ref_count > 0) {
        evict_it->second.pending_delete = true;
        continue;
      }
      return erase_snapshot(evict_it);
    }
    return -1;
  }

  const int32_t checkpoint_slot_id = free_checkpoint_slots_.back();
  free_checkpoint_slots_.pop_back();
#ifndef NDEBUG
  const size_t free_state_index = checkpoint_slot_id - live_slots_;
  DCHECK_LT(free_state_index, checkpoint_slot_free_.size());
  DCHECK_EQ(checkpoint_slot_free_[free_state_index], 1);
  checkpoint_slot_free_[free_state_index] = 0;
#endif
  return checkpoint_slot_id;
}

}  // namespace xllm
