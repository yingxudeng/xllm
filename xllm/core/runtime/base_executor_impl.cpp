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

#include "base_executor_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "util/tensor_helper.h"

namespace xllm {

namespace {

bool should_debug_pure_eager_inputs() { return !FLAGS_enable_graph; }

void debug_log_tensor(const torch::Tensor& tensor,
                      const std::string& name,
                      int num = 16,
                      bool print_value = true) {
  if (!should_debug_pure_eager_inputs()) {
    return;
  }
  xllm::print_tensor(tensor, name, num, true, print_value);
}

void debug_log_model_input_params(const ModelInputParams& params,
                                  const std::string& stage) {
  if (!should_debug_pure_eager_inputs()) {
    return;
  }
  LOG(INFO) << "[force_graph_eager debug] " << stage
            << ", batch_forward_type: " << params.batch_forward_type.to_string()
            << ", num_sequences: " << params.num_sequences
            << ", kv_max_seq_len: " << params.kv_max_seq_len
            << ", q_max_seq_len: " << params.q_max_seq_len
            << ", kv_seq_lens_vec: " << params.kv_seq_lens_vec
            << ", q_seq_lens_vec: " << params.q_seq_lens_vec;
  params.print();
}

}  // namespace

BaseExecutorImpl::BaseExecutorImpl(CausalLM* model,
                                   const ModelArgs& args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {}

ForwardInput BaseExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput BaseExecutorImpl::run(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  std::vector<KVCache>& kv_caches,
                                  const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  if (should_debug_pure_eager_inputs()) {
    LOG(INFO) << "[force_graph_eager debug] BaseExecutorImpl::run"
              << ", enable_graph: " << FLAGS_enable_graph;
    debug_log_tensor(tokens, "BaseExecutorImpl::run tokens");
    debug_log_tensor(positions, "BaseExecutorImpl::run positions");
    debug_log_model_input_params(params, "BaseExecutorImpl::run input params");
    LOG(INFO)
        << "[force_graph_eager debug] BaseExecutorImpl::run using input params "
           "as params_for_capture equivalent because enable_graph=false";
    debug_log_model_input_params(params,
                                 "BaseExecutorImpl::run params_for_capture");
  }
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm
