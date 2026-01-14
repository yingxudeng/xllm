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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/rec_model_utils.h"
#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "platform/device.h"
#include "platform/stream.h"
#include "rec_worker_impl.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

// ConcurrentRecWorkerImpl: Rec Worker supporting multi-stream parallel
// execution Inherits from RecWorkerImpl, adds support for multiple model
// instances and execute stream pool
class ConcurrentRecWorkerImpl : public RecWorkerImpl {
 public:
  // execute_stream_num: execution parallelism, determines the number of model
  // instances and execute streams
  explicit ConcurrentRecWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options);

  ~ConcurrentRecWorkerImpl() override {
    // Release model_ and model_executor_ in destructor to avoid double deletion
    // Ownership actually belongs to model_instances_[0] and
    // executor_instances_[0]
    model_.release();
    model_executor_.release();
    work_pipeline_.release();
  }

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  // Override load_model to load weights for all model instances
  void load_model(std::unique_ptr<ModelLoader> loader) override;

  // Override step_async to support multi-threaded parallel execution
  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 protected:
  // Concurrent versions of pipelines that use thread-specific instances
  class ConcurrentLlmRecPureDevicePipeline final
      : public LlmRecPureDevicePipeline {
   public:
    ConcurrentLlmRecPureDevicePipeline(ConcurrentRecWorkerImpl& worker)
        : LlmRecPureDevicePipeline(worker), concurrent_worker_(worker) {}

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    void warmup(const ForwardInput& inputs);

   protected:
    friend class ConcurrentRecWorkerImpl;
    ConcurrentRecWorkerImpl& concurrent_worker_;

    std::unique_ptr<Stream> stream_;
    std::unique_ptr<CausalLM> model_;
    std::unique_ptr<Executor> executor_;
    std::unique_ptr<ModelContext> context_;
#if defined(USE_CUDA)
    layer::flashinfer::FlashinferWorkspace flashinfer_workspace_;
#endif
  };

  // Factory method to create concurrent pipeline
  static std::unique_ptr<ConcurrentLlmRecPureDevicePipeline>
  create_concurrent_pipeline(RecPipelineType type,
                             ConcurrentRecWorkerImpl& worker);

  // Execution parallelism (number of model instances and execute streams)
  uint32_t max_concurrency_;

  // Independent ThreadPool dedicated to parallel execution of step()
  std::unique_ptr<ThreadPool> step_threadpool_;

  std::vector<std::unique_ptr<ConcurrentLlmRecPureDevicePipeline>>
      multi_stream_pipelines_;

  moodycamel::BlockingConcurrentQueue<size_t> index_queue_;

  // Mutex protecting the allocation process
  std::mutex allocation_mutex_;

  // Update last_step_output (because the base class's update_last_step_output
  // is private)
  void update_last_step_output(const std::optional<ForwardOutput>& output);

  void warmup(const ForwardInput& inputs);
  std::unordered_set<size_t> warmup_set_;
  std::mutex warmup_mutex_;
};

}  // namespace xllm
