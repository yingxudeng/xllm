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

void copy_ssm_slot(const torch::Tensor& ssm_cache,
                   int32_t dst_slot_id,
                   int32_t src_slot_id,
                   int64_t checkpoint_stride) {
  const int64_t dst_offset =
      static_cast<int64_t>(dst_slot_id) * checkpoint_stride;
  const int64_t src_offset =
      static_cast<int64_t>(src_slot_id) * checkpoint_stride;
  ssm_cache.narrow(0, dst_offset, checkpoint_stride)
      .copy_(ssm_cache.narrow(0, src_offset, checkpoint_stride));
}

}  // namespace

LinearStateCheckpointManager::LinearStateCheckpointManager(
    std::vector<KVCache>& kv_caches,
    c10::DeviceIndex device_index)
    : kv_caches_(kv_caches), device_index_(device_index) {}

void LinearStateCheckpointManager::initialize() {
  num_slots_ = 0;
  for (const auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive slots.";
    CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
        << "ssm cache checkpoint layout mismatch, ssm_rows="
        << ssm_cache.size(0) << ", conv_rows=" << conv_cache.size(0);
    if (num_slots_ == 0) {
      num_slots_ = static_cast<int32_t>(conv_cache.size(0));
      continue;
    }
    CHECK_EQ(num_slots_, static_cast<int32_t>(conv_cache.size(0)));
  }

  slot_tags_.assign(num_slots_, LinearStatePrefixHash{});
  LOG(INFO) << "Qwen3.5 linear state checkpoint executor initialized; slots="
            << num_slots_;
}

bool LinearStateCheckpointManager::copy_slot(int32_t dst_slot_id,
                                             int32_t src_slot_id) {
  bool copied = false;
  for (const auto& kv_cache : kv_caches_) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    const int64_t checkpoint_stride = ssm_cache.size(0) / conv_cache.size(0);
    conv_cache.select(0, dst_slot_id).copy_(conv_cache.select(0, src_slot_id));
    copy_ssm_slot(ssm_cache, dst_slot_id, src_slot_id, checkpoint_stride);
    copied = true;
  }
  return copied;
}

std::vector<LinearStateCheckpointManager::RestoreAction>
LinearStateCheckpointManager::restore(
    const std::vector<LinearStateCacheOp>& cache_ops) {
  std::vector<RestoreAction> actions(cache_ops.size(), RestoreAction::SKIPPED);
  if (cache_ops.empty() || num_slots_ == 0) {
    return actions;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  bool restored_any = false;
  for (size_t i = 0; i < cache_ops.size(); ++i) {
    const LinearStateCacheOp& cache_op = cache_ops[i];
    const int32_t live_slot_id = cache_op.linear_state_id;
    const int32_t src_slot_id = cache_op.restore_src_slot_id;
    // No cold-start restore requested for this op. A continued request keeps
    // its warm live slot untouched (has_initial_state stays at the kv-cache
    // default), and a no-prefix cold start has nothing to copy.
    if (src_slot_id < 0) {
      continue;
    }
    if (!slot_in_range(live_slot_id) || !slot_in_range(src_slot_id)) {
      actions[i] = RestoreAction::COLD_START;
      continue;
    }

    // Self-check: the source slot must physically hold the expected checkpoint.
    // Under schedule overlap a save may not have run yet, or the slot may have
    // been reused; either way recompute from scratch instead of copying stale
    // state.
    if (is_zero_prefix_hash(cache_op.restore_prefix_hash) ||
        slot_tags_[src_slot_id] != cache_op.restore_prefix_hash) {
      actions[i] = RestoreAction::COLD_START;
      continue;
    }

    if (!copy_slot(/*dst=*/live_slot_id, /*src=*/src_slot_id)) {
      actions[i] = RestoreAction::COLD_START;
      continue;
    }
    slot_tags_[live_slot_id] = cache_op.restore_prefix_hash;
    actions[i] = RestoreAction::RESTORED;
    restored_any = true;
    VLOG(1) << "Qwen3.5 linear state checkpoint restored; live_slot_id="
            << live_slot_id << ", src_slot_id=" << src_slot_id;
  }
#if defined(USE_NPU)
  if (restored_any) {
    synchronize_current_npu_stream(device_index_, "restore");
  }
#endif
  return actions;
}

void LinearStateCheckpointManager::save(
    const std::vector<LinearStateCacheOp>& cache_ops) {
  if (cache_ops.empty() || num_slots_ == 0) {
    return;
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
    const int32_t live_slot_id = cache_op.linear_state_id;
    const int32_t dst_slot_id = cache_op.save_dst_slot_id;
    if (dst_slot_id < 0 || is_zero_prefix_hash(cache_op.save_prefix_hash)) {
      continue;
    }
    if (!slot_in_range(live_slot_id) || !slot_in_range(dst_slot_id)) {
      continue;
    }
    // The scheduler may reuse an existing checkpoint slot for an unchanged
    // hash; copying a slot onto itself is a no-op but the tag stays correct.
    if (dst_slot_id != live_slot_id &&
        !copy_slot(/*dst=*/dst_slot_id, /*src=*/live_slot_id)) {
      continue;
    }
    slot_tags_[dst_slot_id] = cache_op.save_prefix_hash;
    saved_any = true;
    VLOG(1) << "Qwen3.5 linear state checkpoint saved; live_slot_id="
            << live_slot_id << ", dst_slot_id=" << dst_slot_id;
  }
#if defined(USE_NPU)
  if (saved_any) {
    synchronize_current_npu_stream(device_index_, "save");
  }
#endif
}

}  // namespace xllm
