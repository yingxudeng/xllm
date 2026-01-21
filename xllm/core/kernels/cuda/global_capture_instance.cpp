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
#include "global_capture_instance.h"

#include <glog/logging.h>

#include "attention_runner.h"
#include "piecewise_graphs.h"

namespace xllm::runtime::cuda {

// Static member definitions for thread-local instance management
std::unordered_map<std::thread::id, std::unique_ptr<GlobalCaptureInstance>>
    GlobalCaptureInstance::instances_;
std::mutex GlobalCaptureInstance::instances_mutex_;

GlobalCaptureInstance& GlobalCaptureInstance::get_instance() {
  std::thread::id tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> lock(instances_mutex_);

  auto it = instances_.find(tid);
  if (it == instances_.end()) {
    // Lazy create instance for this thread
    auto [new_it, inserted] = instances_.emplace(
        tid,
        std::unique_ptr<GlobalCaptureInstance>(new GlobalCaptureInstance()));
    return *new_it->second;
  }
  return *it->second;
}

GlobalCaptureInstance::GlobalCaptureInstance() = default;
GlobalCaptureInstance::~GlobalCaptureInstance() = default;

void GlobalCaptureInstance::cleanup_capture_state() {
  is_capturing_ = false;
  current_graph_.reset();
  current_piecewise_graph_.reset();
}

void GlobalCaptureInstance::begin_capture(
    const decltype(at::cuda::graph_pool_handle())& pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(!is_capturing_) << "Already capturing, call end_capture() first";

  is_capturing_ = true;
  graph_pool_ = pool;

  // Reset current_piecewise_graph_
  current_piecewise_graph_ = std::make_unique<PiecewiseGraphs>();

  // Create first graph and begin capture
  current_graph_ = std::make_unique<at::cuda::CUDAGraph>();
  current_graph_->capture_begin(pool, cudaStreamCaptureModeThreadLocal);

  LOG(INFO) << "GlobalCaptureInstance::begin_capture()";
}

void GlobalCaptureInstance::split_graph() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  // split_graph = temporarily_end_graph + temporarily_begin_graph (atomic)
  temporarily_end_graph_locked();
  temporarily_begin_graph_locked();
}

std::unique_ptr<PiecewiseGraphs> GlobalCaptureInstance::end_capture() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_graph_)
      << "Current graph is null, cannot end without active graph. "
      << "Did you call temporarily_end_graph() without "
         "temporarily_begin_graph()?";

  // End last graph and add to piecewise_graph
  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  is_capturing_ = false;

  LOG(INFO) << "GlobalCaptureInstance::end_capture(), total graphs: "
            << current_piecewise_graph_->size()
            << ", total runners: " << current_piecewise_graph_->num_runners();

  return std::move(current_piecewise_graph_);
}

void GlobalCaptureInstance::temporarily_end_graph() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_end_graph_locked();
}

void GlobalCaptureInstance::temporarily_begin_graph() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  temporarily_begin_graph_locked();
}

void GlobalCaptureInstance::temporarily_end_graph_locked() {
  // Must be called with mutex_ held
  CHECK(current_graph_) << "Current graph is null, cannot end. "
                        << "Did you call temporarily_end_graph() twice?";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  // End current graph capture and add to piecewise_graph
  current_graph_->capture_end();
  current_piecewise_graph_->add_graph(std::move(current_graph_));

  LOG(INFO) << "GlobalCaptureInstance::temporarily_end_graph(), total graphs: "
            << current_piecewise_graph_->size();
}

void GlobalCaptureInstance::temporarily_begin_graph_locked() {
  // Must be called with mutex_ held
  CHECK(!current_graph_)
      << "Current graph already exists, cannot begin new graph. "
      << "Did you call temporarily_begin_graph() twice?";

  // Create new graph and begin capture
  current_graph_ = std::make_unique<at::cuda::CUDAGraph>();
  current_graph_->capture_begin(graph_pool_);

  LOG(INFO) << "GlobalCaptureInstance::temporarily_begin_graph()";
}

void GlobalCaptureInstance::register_attention_runner(
    ::xllm::kernel::cuda::AttentionRunner&& runner) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(is_capturing_) << "Not capturing, call begin_capture() first";
  CHECK(current_piecewise_graph_) << "Current piecewise graph is null";

  current_piecewise_graph_->add_attention_runner(std::move(runner));
  LOG(INFO)
      << "GlobalCaptureInstance::register_attention_runner(), total runners: "
      << current_piecewise_graph_->num_runners();
}

}  // namespace xllm::runtime::cuda
