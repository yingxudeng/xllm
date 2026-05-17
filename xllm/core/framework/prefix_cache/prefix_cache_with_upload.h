#pragma once

#include <glog/logging.h>

#include "prefix_cache.h"
#include "util/double_buffer.h"

namespace xllm {
class PrefixCacheWithUpload final : public PrefixCache {
 public:
  explicit PrefixCacheWithUpload(uint32_t block_size);

  ~PrefixCacheWithUpload();

  // insert the token ids and blocks into the prefix tree
  // and set hash key to the corresponding block
  // return the length of new inserted tokens
  size_t insert(const Slice<int32_t>& token_ids,
                std::vector<Block>& blocks,
                size_t existed_shared_blocks_num) override;

  // insert the blocks with hash key into the prefix tree
  size_t insert(const std::vector<Block>& blocks) override;
  size_t insert(Slice<Block>& blocks) override;

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  size_t evict(size_t n_blocks) override;

  // insert with checkpoint slot IDs for linear state
  size_t insert_with_checkpoint_slots(
      const Slice<int32_t>& token_ids,
      std::vector<Block>& blocks,
      size_t existed_shared_blocks_num,
      const Slice<int32_t>& checkpoint_slot_ids) override;

  // evict blocks and collect freed checkpoint slot IDs
  size_t evict_with_freed_slots(
      size_t n_blocks,
      std::vector<int32_t>& freed_checkpoint_slots) override;

  virtual KvCacheEvent* get_upload_kvcache_events() override;

 private:
  void save_event_async(const bool is_insert, std::vector<XXH3Key>& keys);

 private:
  ThreadPool threadpool_;

  std::mutex mutex_;
  DoubleBuffer<KvCacheEvent> db_kvcache_events_;
};

}  // namespace xllm
