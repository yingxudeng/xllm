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

#include "npu_profiler.h"

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <filesystem>
#include <string>

namespace xllm {

namespace py = pybind11;

// pybind11 types are declared with hidden visibility; matching Impl's
// visibility avoids GCC's -Wattributes warning under -Werror when the outer
// NpuProfiler class has default visibility.
struct __attribute__((visibility("hidden"))) NpuProfiler::Impl {
  py::object profiler;
};

NpuProfiler& NpuProfiler::get_instance() {
  static NpuProfiler instance;
  return instance;
}

NpuProfiler::NpuProfiler() : impl_(std::make_unique<Impl>()) {}

NpuProfiler::~NpuProfiler() {
  if (impl_ && impl_->profiler) {
    try {
      py::gil_scoped_acquire gil;
      impl_->profiler.release().dec_ref();
    } catch (...) {
      // Interpreter may already be finalized; leak the reference in that case
      // rather than aborting during static teardown.
    }
  }
}

bool NpuProfiler::is_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

bool NpuProfiler::start(const std::string& profile_dir, int32_t rank) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    LOG(WARNING) << "NpuProfiler::start called while already running; ignoring";
    return true;
  }

  const std::string resolved_dir =
      profile_dir.empty() ? std::filesystem::current_path().string()
                          : profile_dir;
  std::error_code ec;
  std::filesystem::create_directories(resolved_dir, ec);
  if (ec) {
    LOG(ERROR) << "NpuProfiler: failed to create profile dir " << resolved_dir
               << ": " << ec.message();
    return false;
  }

  const std::string worker_name = "xllm_rank" + std::to_string(rank);

  try {
    py::gil_scoped_acquire gil;
    py::module_ tp = py::module_::import("torch_npu.profiler");

    if (!impl_->profiler) {
      // Level1 + AiCoreNone is the cheapest CANN AclProfiler setting that
      // still yields op_summary.csv and kernel_details.csv. The remaining
      // fields are all pinned to their conservative defaults so the trace
      // envelope stays stable across runs.
      py::object experimental = tp.attr("_ExperimentalConfig")(
          py::arg("export_type") = tp.attr("ExportType").attr("Text"),
          py::arg("profiler_level") = tp.attr("ProfilerLevel").attr("Level1"),
          py::arg("msprof_tx") = false,
          py::arg("aic_metrics") = tp.attr("AiCMetrics").attr("AiCoreNone"),
          py::arg("l2_cache") = false,
          py::arg("op_attr") = false,
          py::arg("data_simplification") = true,
          py::arg("record_op_args") = false,
          py::arg("gc_detect_threshold") = py::none());

      py::list activities;
      activities.append(tp.attr("ProfilerActivity").attr("CPU"));
      activities.append(tp.attr("ProfilerActivity").attr("NPU"));

      py::object trace_handler = tp.attr("tensorboard_trace_handler")(
          py::str(resolved_dir), py::arg("worker_name") = py::str(worker_name));

      impl_->profiler =
          tp.attr("profile")(py::arg("activities") = activities,
                             py::arg("with_stack") = false,
                             py::arg("profile_memory") = false,
                             py::arg("with_modules") = false,
                             py::arg("experimental_config") = experimental,
                             py::arg("on_trace_ready") = trace_handler);
    }

    impl_->profiler.attr("start")();
  } catch (const std::exception& e) {
    LOG(ERROR) << "NpuProfiler::start failed: " << e.what();
    return false;
  }

  running_ = true;
  LOG(INFO) << "NpuProfiler started: dir=" << resolved_dir
            << ", worker_name=" << worker_name;
  return true;
}

bool NpuProfiler::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    LOG(WARNING) << "NpuProfiler::stop called while not running; ignoring";
    return true;
  }

  try {
    py::gil_scoped_acquire gil;
    impl_->profiler.attr("stop")();
  } catch (const std::exception& e) {
    LOG(ERROR) << "NpuProfiler::stop failed: " << e.what();
    running_ = false;
    return false;
  }

  running_ = false;
  LOG(INFO) << "NpuProfiler stopped. Trace written by "
               "tensorboard_trace_handler.";
  return true;
}

}  // namespace xllm
