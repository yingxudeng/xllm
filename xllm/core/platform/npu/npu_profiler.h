/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>
#include <mutex>
#include <string>

namespace xllm {

// NpuProfiler drives online timeline collection through
// torch_npu.profiler.profile. Unlike the in-process Kineto path
// (TorchProfiler), this backend routes directly to CANN AclProfiler and
// therefore produces artifacts under a *_ascend_pt/ directory containing
// ASCEND_PROFILER_OUTPUT/*.csv, which is what torch_npu.profiler.profiler
// .analyse() consumes and what MindStudio Insight expects.
//
// Like TorchProfiler this is a process-wide singleton. xLLM runs one worker
// thread per device inside a single process; the underlying Python profiler
// object is created lazily on the first start() and kept alive across
// subsequent start/stop cycles so trace_name (which encodes rank and is
// baked into on_trace_ready) does not change mid-run.
//
// IMPORTANT: NPU CPU-op capture uses thread-local RecordFunction callbacks,
// so start() and stop() must be invoked on the compute thread that runs the
// forward pass — WorkerImpl::threadpool_ (single-thread) is that thread.
class NpuProfiler {
 public:
  static NpuProfiler& get_instance();

  // Open the CANN AclProfiler collection window. Repeated calls while already
  // running are ignored. Returns true on success (or if already running).
  bool start(const std::string& profile_dir, int32_t rank);

  // Close the collection window and let torch_npu's tensorboard_trace_handler
  // dump the *_ascend_pt/ directory under `profile_dir`. Calling stop() while
  // not running is a no-op that returns true.
  bool stop();

  bool is_running() const;

 private:
  NpuProfiler();
  ~NpuProfiler();
  NpuProfiler(const NpuProfiler&) = delete;
  NpuProfiler& operator=(const NpuProfiler&) = delete;

  // Impl holds the cached pybind11::object; kept out of the header so callers
  // do not need pybind11 include paths and so the class visibility does not
  // conflict with pybind11's hidden-visibility types under -Wattributes.
  struct Impl;
  std::unique_ptr<Impl> impl_;

  mutable std::mutex mutex_;
  bool running_ = false;
};

}  // namespace xllm
