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
#include <ATen/cuda/CUDAGraph.h>

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace xllm::kernel::cuda {
// Forward declaration
class AttentionRunner;
}  // namespace xllm::kernel::cuda

namespace xllm::runtime::cuda {
// Forward declaration - full definition in piecewise_graphs.h
class PiecewiseGraphs;
}  // namespace xllm::runtime::cuda

namespace xllm::runtime::cuda {

// Global CUDA Graph Capture instance management
// Thread-safe: each thread gets its own instance via get_instance()
// Note: Stream management should be handled by the caller (e.g.,
// CudaGraphExecutor)
class GlobalCaptureInstance {
 public:
  // Returns the instance for the current thread (lazy created)
  static GlobalCaptureInstance& get_instance();
  // Begin capture: reset current_piecewise_graph_, create first graph
  void begin_capture(const decltype(at::cuda::graph_pool_handle())& pool);
  // End capture: end last graph, return current_piecewise_graph_
  std::unique_ptr<PiecewiseGraphs> end_capture();
  // End current graph capture, add to current_piecewise_graph_
  void temporarily_end_graph();
  // Create new current_graph_ and begin capture
  void temporarily_begin_graph();
  // Split graph: temporarily_end_graph() + temporarily_begin_graph()
  void split_graph();
  // Register attention runner to current_piecewise_graph_
  void register_attention_runner(
      ::xllm::kernel::cuda::AttentionRunner&& runner);

  // Check if currently capturing
  bool is_capturing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_capturing_;
  }
  // Get current graph (for use in hooks)
  at::cuda::CUDAGraph* get_current_graph() { return current_graph_.get(); }

 private:
  // Allow unique_ptr to call destructor
  friend struct std::default_delete<GlobalCaptureInstance>;

  // Constructor and destructor must be defined in .cpp where PiecewiseGraphs
  // is complete
  GlobalCaptureInstance();
  ~GlobalCaptureInstance();

  // Helper: cleanup state
  void cleanup_capture_state();

  // Internal versions without locking (must be called with mutex_ held)
  void temporarily_end_graph_locked();
  void temporarily_begin_graph_locked();

  bool is_capturing_ = false;
  std::unique_ptr<at::cuda::CUDAGraph> current_graph_;
  std::unique_ptr<PiecewiseGraphs> current_piecewise_graph_;
  decltype(at::cuda::graph_pool_handle()) graph_pool_;
  mutable std::mutex mutex_;  // Protect capture state

  // Static members for thread-local instance management
  static std::unordered_map<std::thread::id,
                            std::unique_ptr<GlobalCaptureInstance>>
      instances_;
  static std::mutex instances_mutex_;
};
}  // namespace xllm::runtime::cuda
