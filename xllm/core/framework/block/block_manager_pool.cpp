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

#include "block_manager_pool.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "block_manager_impl.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "framework/model/model_input_params.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"

namespace xllm {

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  single_block_managers_.reserve(dp_size);
  const uint32_t default_max_single_block_sequences =
      options_.max_concurrent_requests() > 0
          ? options_.max_concurrent_requests()
          : static_cast<uint32_t>(std::max(
                ::xllm::ServiceConfig::get_instance().max_concurrent_requests(),
                0));

  BlockManager::Options block_options;
  block_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.host_num_blocks() > 0
                               ? false
                               : options_.enable_cache_upload())
      .sliding_window_size(options_.sliding_window_size())
      .swa_blocks_per_seq(options_.swa_blocks_per_seq())
      .max_tokens_per_batch(options_.max_tokens_per_batch())
      .manager_types(options_.manager_types())
      .compress_ratios(options_.compress_ratios())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .hasher_type(options_.hasher_type());

  uint32_t num_single_blocks = std::max<uint32_t>(
      options_.num_single_blocks(), default_max_single_block_sequences + 2);
  if (options_.enable_linear_state()) {
    const bool has_explicit_single_block_capacity =
        options_.single_block_capacity() > 0;
    const uint32_t default_single_block_capacity =
        options_.max_seqs_per_batch() > 0
            ? options_.max_seqs_per_batch() + 2
            : default_max_single_block_sequences + 2;
    const uint32_t single_block_capacity =
        has_explicit_single_block_capacity ? options_.single_block_capacity()
                                           : default_single_block_capacity;
    num_single_blocks =
        has_explicit_single_block_capacity
            ? single_block_capacity
            : std::max<uint32_t>(num_single_blocks, single_block_capacity);
  }
  CHECK_GT(num_single_blocks, 0u) << "num_single_blocks must be positive";

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options_.enable_xtensor()) {
      // Use XTensorBlockManagerImpl for xtensor mode.
      CHECK_GT(options_.num_layers(), 0)
          << "num_layers must be set when enable_xtensor is true";
      CHECK_GT(options_.slot_size(), 0)
          << "slot_size must be set when enable_xtensor is true";
      size_t page_size =
          ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
      // In the current implementation, K and V must be the same size, so we
      // divide by 2.
      size_t block_mem_size =
          static_cast<size_t>(options_.block_size()) * options_.slot_size() / 2;
      block_managers_.emplace_back(
          std::make_unique<XTensorBlockManagerImpl>(block_options,
                                                    options_.num_layers(),
                                                    block_mem_size,
                                                    page_size,
                                                    /*dp_rank=*/i,
                                                    options_.model_id()));
    } else if (!options_.manager_types().empty()) {
      block_managers_.emplace_back(
          std::make_unique<CompositeBlockManager>(block_options));
    } else if (options_.enable_disagg_pd() || options_.enable_kvcache_store()) {
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(block_options));
    } else {
      block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(block_options));
    }
    // Scheduler-side per-sequence resources share one logical single-block
    // pool. Worker-side embedding and linear-state caches remain physically
    // separate and are addressed via transport fields.
    single_block_managers_.emplace_back(std::make_unique<SingleBlockManager>(
        /*num_blocks=*/num_single_blocks,
        /*resource_name=*/"single block",
        /*exhaustion_message=*/"No more single-block ids available"));
  }

  if (options_.enable_linear_state()) {
    CHECK_GT(options_.linear_state_num_slots(), 0)
        << "linear_state_num_slots must be set when linear state is enabled";
    linear_state_prefix_caches_.reserve(dp_size);
    for (int32_t i = 0; i < dp_size; ++i) {
      linear_state_prefix_caches_.emplace_back(
          std::make_unique<LinearStatePrefixCache>(
              options_.linear_state_num_slots()));
    }
  }

  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(block_managers_.size());
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  if (block_managers_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = block_managers_[0]->num_free_blocks();
  if (options_.enable_linear_state() &&
      single_block_managers_[0]->num_free_blocks() == 0) {
    max_free = 0;
  }

  for (size_t i = 1; i < block_managers_.size(); ++i) {
    if (options_.enable_linear_state() &&
        single_block_managers_[i]->num_free_blocks() == 0) {
      continue;
    }
    const size_t current_free = block_managers_[i]->num_free_blocks();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

bool BlockManagerPool::allocate_single_block(Sequence* sequence,
                                             int32_t dp_rank) {
  CHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  if (sequence->has_single_block_id()) {
    // Both per-sequence slots are acquired together; if one is held the other
    // must be too.
    return true;
  }

  auto single_blocks = single_block_managers_[dp_rank]->allocate(1);
  if (single_blocks.empty()) {
    const auto* manager = single_block_managers_[dp_rank].get();
    LOG(ERROR) << "Failed to allocate single block! dp_rank=" << dp_rank
               << ", free=" << manager->num_free_blocks()
               << ", used=" << manager->num_used_blocks()
               << ", total=" << manager->num_total_blocks()
               << ", max_seqs_per_batch=" << options_.max_seqs_per_batch()
               << ", configured_single_blocks=" << options_.num_single_blocks();
    return false;
  }
  sequence->set_single_block(std::move(single_blocks[0]));

  if (!allocate_linear_state_slot(sequence, dp_rank)) {
    // Roll back the single block so the pair stays consistent.
    auto single_block = sequence->reset_single_block();
    single_block_managers_[dp_rank]->deallocate({&single_block, 1});
    return false;
  }
  return true;
}

void BlockManagerPool::deallocate_single_block(Sequence* sequence,
                                               int32_t dp_rank) {
  DCHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  release_linear_state_slot(sequence);
  auto single_block = sequence->reset_single_block();
  if (!single_block.is_valid()) {
    return;
  }
  single_block_managers_[dp_rank]->deallocate({&single_block, 1});
}

bool BlockManagerPool::allocate_linear_state_slot(Sequence* sequence,
                                                  int32_t dp_rank) {
  if (!options_.enable_linear_state()) {
    return true;
  }
  CHECK_LT(static_cast<size_t>(dp_rank), linear_state_prefix_caches_.size());
  if (sequence->has_linear_state_slot()) {
    return true;
  }
  Block slot = linear_state_prefix_caches_[dp_rank]->allocate_live_slot();
  if (!slot.is_valid()) {
    LOG(ERROR) << "Failed to acquire linear state slot!";
    return false;
  }
  sequence->set_linear_state_slot(std::move(slot));
  return true;
}

void BlockManagerPool::release_linear_state_slot(Sequence* sequence) {
  if (!options_.enable_linear_state()) {
    return;
  }
  // Dropping the handle returns the live slot to its origin pool (the Block
  // carries its own manager pointer), so no dp_rank routing is needed here.
  // Boundary checkpoints made while running already captured any reusable
  // state.
  sequence->reset_linear_state_slot();
}

LinearStatePrefixCache* BlockManagerPool::linear_state_prefix_cache(
    int32_t dp_rank) {
  if (!options_.enable_linear_state()) {
    return nullptr;
  }
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), linear_state_prefix_caches_.size());
  return linear_state_prefix_caches_[dp_rank].get();
}

LinearStateCheckpointReservations
BlockManagerPool::resolve_linear_state_cache_ops(
    int32_t dp_rank,
    std::vector<LinearStateCacheOp>* cache_ops,
    const std::vector<Sequence*>& sequences) {
  LinearStatePrefixCache* cache = linear_state_prefix_cache(dp_rank);
  LinearStateCheckpointReservations checkpoint_reservations;
  if (cache != nullptr && cache_ops != nullptr && !cache_ops->empty()) {
    checkpoint_reservations = cache->resolve_cache_ops(cache_ops, sequences);
  }
  // dp_rank is a pool-level routing tag, not a per-cache concept: stamp it here
  // (the sole owner of the per-dp-rank cache lookup) so that
  // commit_linear_state_reservations can route the reservation back to the same
  // cache after the forward.
  checkpoint_reservations.dp_rank_ = dp_rank;
  return checkpoint_reservations;
}

void BlockManagerPool::commit_linear_state_reservations(
    LinearStateCheckpointReservations&& checkpoint_reservations) {
  if (checkpoint_reservations.dp_rank_ < 0) {
    return;
  }
  LinearStatePrefixCache* cache =
      linear_state_prefix_cache(checkpoint_reservations.dp_rank_);
  if (cache == nullptr) {
    return;
  }
  cache->commit_reservations(std::move(checkpoint_reservations));
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    deallocate(sequence);
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);

  if (block_managers_[dp_rank]->is_composite()) {
    // TODO: not supporte prefix cache for composite manager yet.
    block_managers_[dp_rank]->deallocate_sequence(sequence);
    deallocate_single_block(sequence, dp_rank);
    sequence->reset();
    return;
  }

  // add blocks to the prefix cache
  cache(sequence);
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_single_block(sequence, dp_rank);
  // release the blocks after prefix cache insertion
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

bool BlockManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence, sequence->num_tokens())) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  const bool started_empty = sequence->kv_state().num_kv_blocks() == 0;
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    return false;
  }

  if (block_managers_[dp_rank]->is_composite()) {
    // TODO: not supporte prefix cache for composite manager yet.
    if (!block_managers_[dp_rank]->allocate_for_sequence(sequence,
                                                         num_tokens)) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      return false;
    }
    return true;
  }

  // first try to allocate shared blocks
  if (started_empty) {
    BlockManagerPool::allocate_shared(sequence);
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    return process_beam_search(sequence, /*need_swap*/ true);
  }
  process_beam_search(sequence);

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    if (started_empty) {
      block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      sequence->reset();
    }
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";

    return false;
  }

  sequence->add_kv_blocks(blocks);

  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    return false;
  }
  if (block_managers_[dp_rank]->is_composite()) {
    if (!block_managers_[dp_rank]->allocate_for_sequence(
            sequence, sequence->num_tokens())) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      return false;
    }
    return true;
  }

  std::vector<Block> shared_blocks;
  size_t shared_num = 0;
  if (options_.enable_prefix_cache()) {
    sequence->update_block_hashes(static_cast<uint32_t>(options_.block_size()),
                                  options_.hasher_type());
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    shared_blocks =
        block_managers_[dp_rank]->allocate_shared(sequence->tokens(),
                                                  existed_shared_blocks,
                                                  sequence->mm_data(),
                                                  sequence->block_hashes());
    trim_shared_blocks_to_linear_state(
        dp_rank,
        sequence,
        sequence->kv_state().shared_kv_blocks_num(),
        &shared_blocks);
    if (!shared_blocks.empty()) {
      sequence->add_kv_blocks(shared_blocks);
      sequence->kv_state().incr_shared_kv_blocks_num(shared_blocks.size());
      shared_num = shared_blocks.size();
    }
  }

  const size_t block_size = options_.block_size();
  size_t num_tokens = sequence->tokens().size() - shared_num * block_size;

  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > 0) {
    const auto blocks = block_managers_[dp_rank]->allocate(num_blocks_needed);
    if (blocks.size() != num_blocks_needed) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      if (shared_num != 0) {
        block_managers_[dp_rank]->deallocate(shared_blocks);
        sequence->reset();
      }
      return false;
    }

    sequence->add_kv_blocks(std::move(blocks));
  }

  sequence->kv_state().incr_kv_cache_tokens_num(sequence->tokens().size());
  return true;
}

bool BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return true;
  }

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    auto new_blocks = block_managers_[dp_rank]->allocate(1);
    if (new_blocks.size() == 0) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks[0]);
  } else {
    sequence->kv_state().process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    int32_t dp_rank = get_dp_rank(sequence);
    sequence->update_block_hashes(static_cast<uint32_t>(options_.block_size()),
                                  options_.hasher_type());
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    std::vector<Block> shared_blocks =
        block_managers_[dp_rank]->allocate_shared(sequence->tokens(),
                                                  existed_shared_blocks,
                                                  sequence->mm_data(),
                                                  sequence->block_hashes());
    trim_shared_blocks_to_linear_state(
        dp_rank,
        sequence,
        sequence->kv_state().shared_kv_blocks_num(),
        &shared_blocks);
    sequence->add_shared_kv_blocks(std::move(shared_blocks));
  }
}

void BlockManagerPool::cache(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  if (block_managers_[dp_rank]->is_composite()) {
    // Prefix cache is not supported for CompositeBlockManager yet.
    return;
  }
  sequence->update_block_hashes(static_cast<uint32_t>(options_.block_size()),
                                options_.hasher_type());
  const auto token_ids = sequence->cached_tokens();
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  auto existed_shared_blocks_num = sequence->kv_state().shared_kv_blocks_num();
  block_managers_[dp_rank]->cache(token_ids,
                                  *blocks,
                                  existed_shared_blocks_num,
                                  sequence->mm_data(),
                                  sequence->block_hashes());
}

void BlockManagerPool::cache(Sequence* sequence, size_t num_tokens) {
  CHECK(sequence != nullptr);
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  if (block_managers_[dp_rank]->is_composite()) {
    // Prefix cache is not supported for CompositeBlockManager yet.
    return;
  }

  // Only publish the full blocks that are guaranteed to be computed: clamp the
  // requested token budget to the blocks actually allocated and to the
  // sequence's own tokens. The last partial block is dropped by the prefix
  // cache insert (it aligns to block boundary).
  const size_t block_size = static_cast<size_t>(options_.block_size());
  const size_t available_tokens_num =
      std::min({num_tokens,
                sequence->kv_state().num_kv_blocks() * block_size,
                sequence->tokens().size()});
  const size_t existed_shared_blocks_num =
      sequence->kv_state().shared_kv_blocks_num();
  if (available_tokens_num <= existed_shared_blocks_num * block_size) {
    return;
  }

  // Block hashes are already computed during allocate(); this is an idempotent
  // no-op kept for safety so we do not depend on allocate() running first.
  sequence->update_block_hashes(static_cast<uint32_t>(options_.block_size()),
                                options_.hasher_type());
  const auto token_ids = sequence->tokens().slice(0, available_tokens_num);
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  CHECK_GE(blocks->size(), existed_shared_blocks_num);
  block_managers_[dp_rank]->cache(token_ids,
                                  *blocks,
                                  existed_shared_blocks_num,
                                  sequence->mm_data(),
                                  sequence->block_hashes());
}

void BlockManagerPool::get_merged_kvcache_event(KvCacheEvent* event) const {
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    block_managers_[i]->get_merged_kvcache_event(event);
  }
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  float perc = 0.0;
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    perc += block_managers_[i]->kv_cache_utilization();
  }
  return perc / block_managers_.size();
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(block_managers_.size());
  if (!options_.enable_prefix_cache()) {
    return num_blocks_in_prefix_cache;
  }
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    if (block_managers_[dp_rank]->is_composite()) {
      // CompositeBlockManager does not support prefix-cache stats yet.
      num_blocks_in_prefix_cache[dp_rank] = 0;
      continue;
    }
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_free_blocks[dp_rank] = block_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_used_blocks[dp_rank] = block_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return block_managers_[dp_rank]->kv_cache_utilization();
}

void BlockManagerPool::trim_shared_blocks_to_linear_state(
    int32_t dp_rank,
    Sequence* sequence,
    size_t existed_shared_blocks_num,
    std::vector<Block>* shared_blocks) {
  if (!options_.enable_linear_state() || shared_blocks->empty()) {
    return;
  }
  CHECK(sequence != nullptr);

  // Limit KV prefix reuse to the longest prefix whose block-aligned boundary
  // still has a linear-state checkpoint in the prefix cache, so a reused prefix
  // always has restorable recurrent state. Checkpoints already known from a
  // prior step (existed_shared_blocks_num) stay safe.
  //
  // Additionally cap new reuse so at least one token is left for the current
  // forward. Without it, an exact-prompt match could later be popped one block
  // back (see KVCacheState::add_shared_kv_blocks) onto a non-checkpoint
  // boundary, leaving the sequence unable to restore its recurrent state.
  CHECK_LT(static_cast<size_t>(dp_rank), linear_state_prefix_caches_.size());
  LinearStatePrefixCache* pool = linear_state_prefix_caches_[dp_rank].get();
  const int32_t block_size = options_.block_size();
  if (block_size <= 0) {
    return;
  }
  const size_t total_blocks = shared_blocks->size();
  size_t safe_shared_blocks = std::min(existed_shared_blocks_num, total_blocks);
  // Already-held blocks are never trimmed; the cap only bounds new reuse.
  const size_t max_reusable_blocks =
      sequence->num_tokens() == 0
          ? 0
          : (sequence->num_tokens() - 1) / static_cast<size_t>(block_size);
  const size_t reusable_limit =
      std::max(safe_shared_blocks, std::min(total_blocks, max_reusable_blocks));

  const size_t token_blocks =
      sequence->tokens().size() / static_cast<size_t>(block_size);
  const size_t hash_limit = std::min(reusable_limit, token_blocks);
  const std::vector<PrefixHash> hashes = compute_linear_state_prefix_hashes(
      sequence->tokens(),
      static_cast<size_t>(block_size),
      hash_limit * static_cast<size_t>(block_size));
  for (size_t block_idx = safe_shared_blocks; block_idx < hashes.size();
       ++block_idx) {
    if (pool->contains(XXH3Key(hashes[block_idx].data()))) {
      safe_shared_blocks = block_idx + 1;
    }
  }
  CHECK_LE(safe_shared_blocks, total_blocks);
  if (safe_shared_blocks == total_blocks) {
    return;
  }

  block_managers_[dp_rank]->deallocate(
      Slice<Block>(*shared_blocks).slice(safe_shared_blocks));
  shared_blocks->resize(safe_shared_blocks);
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  DCHECK(!block_managers_[dp_rank].get()->is_composite())
      << "Composite manager does not support deallocate_without_cache yet.";

  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_single_block(sequence, dp_rank);
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }

  // Reserve padding block on each XTensorBlockManagerImpl.
  for (auto& manager : block_managers_) {
    auto* xtensor_manager =
        dynamic_cast<XTensorBlockManagerImpl*>(manager.get());
    if (xtensor_manager) {
      xtensor_manager->reserve_xtensor_padding_blocks();
    }
  }

  // Start prealloc thread once (PageAllocator is shared by all managers)
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm
