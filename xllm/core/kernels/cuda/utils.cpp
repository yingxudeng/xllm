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

#include "utils.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <vector>

#include "core/platform/device.h"
#include "core/util/env_var.h"

namespace {
const std::unordered_map<torch::ScalarType, std::string_view>
    filename_safe_dtype_map = {
        {torch::kFloat16, "f16"},
        {torch::kBFloat16, "bf16"},
        {torch::kFloat8_e4m3fn, "e4m3"},
        {torch::kFloat8_e5m2, "e5m2"},
        {torch::kInt8, "i8"},
        {torch::kUInt8, "u8"},
        {torch::kInt32, "i32"},
        {torch::kUInt32, "u32"},
        {torch::kInt64, "i64"},
        {torch::kUInt64, "u64"},
};

// Matches FlashInfer PrefillPlanInfo::ToVector() layout in:
// flashinfer/include/flashinfer/attention/scheduler.cuh
struct FlashinferPrefillPlanInfoView {
  int64_t padded_batch_size = -1;
  int64_t total_num_rows = -1;
  int64_t total_num_rows_offset = -1;
  int64_t cta_tile_q = -1;
  int64_t request_indices_offset = -1;
  int64_t qo_tile_indices_offset = -1;
  int64_t kv_tile_indices_offset = -1;
  int64_t merge_indptr_offset = -1;
  int64_t o_indptr_offset = -1;
  int64_t kv_chunk_size_ptr_offset = -1;
  int64_t v_offset = -1;
  int64_t s_offset = -1;
  int64_t block_valid_mask_offset = -1;
  int64_t enable_cuda_graph = -1;
  int64_t split_kv = -1;
};

inline std::optional<FlashinferPrefillPlanInfoView> parse_plan_info_vec(
    const std::vector<int64_t>& v) {
  // Support both fa2 (9 elements) and fa3 (15 elements) formats
  if (v.size() != 9 && v.size() != 15) return std::nullopt;
  FlashinferPrefillPlanInfoView out;
  // Common fields for both fa2 and fa3 (first 9 elements)
  out.padded_batch_size = v[0];
  out.total_num_rows = v[1];
  out.total_num_rows_offset = v[2];
  out.cta_tile_q = v[3];
  out.request_indices_offset = v[4];
  out.qo_tile_indices_offset = v[5];
  out.kv_tile_indices_offset = v[6];
  out.merge_indptr_offset = v[7];
  out.o_indptr_offset = v[8];
  // Additional fields only in fa3 (elements 9-14)
  if (v.size() == 15) {
    out.kv_chunk_size_ptr_offset = v[9];
    out.v_offset = v[10];
    out.s_offset = v[11];
    out.block_valid_mask_offset = v[12];
    out.enable_cuda_graph = v[13];
    out.split_kv = v[14];
  } else {
    // For fa2, set additional fields to default/invalid values
    out.kv_chunk_size_ptr_offset = -1;
    out.v_offset = -1;
    out.s_offset = -1;
    out.block_valid_mask_offset = -1;
    out.enable_cuda_graph = -1;
    out.split_kv = -1;
  }
  return out;
}
}  // namespace

namespace xllm::kernel::cuda {

bool support_pdl() { return Device::is_enable_pdl(); }

std::string path_to_uri_so_lib(const std::string& uri) {
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask) {
  bool support_fa3_backend =
      (pos_encoding_mode == 0) && !use_fp16_qk_reduction && !use_custom_mask;

  if (Device::is_support_sm90a() && support_fa3_backend) {
    return "fa3";
  }
  return "fa2";
}

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction) {
  std::ostringstream oss;
  oss << "batch_prefill_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False") << "_"
      << "f16qk_" << (use_fp16_qk_reduction ? "True" : "False");

  if (backend == "fa3") oss << "_sm90";

  return oss.str();
}

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap) {
  std::ostringstream oss;
  oss << "batch_decode_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False");

  return oss.str();
}

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  static std::unordered_map<std::string, torch::Tensor> cache_buffer_map;
  int32_t seq_len_pow2 = xllm::util::ceil_pow2(seq_len);

  std::string key = std::string("range_") + std::to_string(seq_len_pow2);
  auto it = cache_buffer_map.find(key);
  if (it != cache_buffer_map.end()) {
    return it->second.slice(0, 0, seq_len);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor buffer = torch::arange(seq_len_pow2, options);
  cache_buffer_map.insert(std::make_pair(key, buffer));
  return buffer.slice(0, 0, seq_len);
}

void debug_log_prefill_inputs(
    const std::string& uri,
    const torch::Tensor& plan_info,
    const torch::Tensor& float_workspace_buffer,
    const torch::Tensor& int_workspace_buffer,
    const torch::Tensor& page_locked_int_workspace_buffer,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& kv_cu_seq_lens,
    int64_t window_left,
    bool is_causal) {
  // Basic tensor info
  LOG(INFO) << "[flashinfer_prefill_debug] uri=" << uri
            << " is_causal=" << is_causal
            << " mask_mode_code=" << (is_causal ? 1 : 0)
            << " enable_cuda_graph(plan?)=" << (plan_info.defined() ? 1 : 0);
  LOG(INFO) << "[flashinfer_prefill_debug] q shape=" << query.sizes()
            << " stride=" << query.strides() << " dtype=" << query.scalar_type()
            << " device=" << query.device();
  LOG(INFO) << "[flashinfer_prefill_debug] k shape=" << key.sizes()
            << " stride=" << key.strides() << " dtype=" << key.scalar_type()
            << " device=" << key.device();
  LOG(INFO) << "[flashinfer_prefill_debug] v shape=" << value.sizes()
            << " stride=" << value.strides() << " dtype=" << value.scalar_type()
            << " device=" << value.device();

  // Workspace info
  const int64_t float_ws_bytes =
      float_workspace_buffer.numel() * float_workspace_buffer.element_size();
  const int64_t int_ws_bytes =
      int_workspace_buffer.numel() * int_workspace_buffer.element_size();
  LOG(INFO) << "[flashinfer_prefill_debug] float_ws bytes=" << float_ws_bytes
            << " int_ws bytes=" << int_ws_bytes
            << " window_left=" << window_left;

  // plan_info decode (best-effort)
  try {
    // Ensure any async planner memcpy has finished before we read workspace on
    // CPU.
    cudaError_t st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
      LOG(WARNING)
          << "[flashinfer_prefill_debug] cudaDeviceSynchronize failed: "
          << cudaGetErrorString(st);
    }
    auto plan_cpu = plan_info.to(torch::kCPU);
    std::vector<int64_t> vec;
    vec.reserve(plan_cpu.numel());
    for (int64_t i = 0; i < plan_cpu.numel(); ++i) {
      vec.push_back(plan_cpu[i].item<int64_t>());
    }
    std::ostringstream oss;
    oss << "[flashinfer_prefill_debug] plan_info_vec(size=" << vec.size()
        << "): [";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i] << (i + 1 == vec.size() ? "" : ", ");
    }
    oss << "]";
    LOG(INFO) << oss.str();

    auto parsed = parse_plan_info_vec(vec);
    if (parsed.has_value()) {
      const auto& p = parsed.value();
      LOG(INFO) << "[flashinfer_prefill_debug] plan: padded_batch_size="
                << p.padded_batch_size << " total_num_rows=" << p.total_num_rows
                << " cta_tile_q=" << p.cta_tile_q
                << " enable_cuda_graph=" << p.enable_cuda_graph
                << " split_kv=" << p.split_kv;
      LOG(INFO) << "[flashinfer_prefill_debug] plan int offsets:"
                << " request_indices=" << p.request_indices_offset
                << " qo_tile_indices=" << p.qo_tile_indices_offset
                << " kv_tile_indices=" << p.kv_tile_indices_offset
                << " o_indptr=" << p.o_indptr_offset
                << " kv_chunk_size_ptr=" << p.kv_chunk_size_ptr_offset
                << " merge_indptr=" << p.merge_indptr_offset
                << " total_num_rows_offset=" << p.total_num_rows_offset
                << " block_valid_mask=" << p.block_valid_mask_offset;
      LOG(INFO) << "[flashinfer_prefill_debug] plan float offsets:"
                << " v_offset=" << p.v_offset << " s_offset=" << p.s_offset;

      // Rough bounds checks (offsets are byte offsets into uint8 workspaces)
      auto check_offset = [&](const char* name, int64_t off, int64_t bytes) {
        if (off < 0) return;
        if (off >= bytes) {
          LOG(ERROR) << "[flashinfer_prefill_debug] workspace OOB risk: "
                     << name << " offset=" << off << " >= ws_bytes=" << bytes;
        }
      };
      check_offset(
          "int.request_indices", p.request_indices_offset, int_ws_bytes);
      check_offset(
          "int.qo_tile_indices", p.qo_tile_indices_offset, int_ws_bytes);
      check_offset(
          "int.kv_tile_indices", p.kv_tile_indices_offset, int_ws_bytes);
      check_offset("int.o_indptr", p.o_indptr_offset, int_ws_bytes);
      check_offset(
          "int.kv_chunk_size_ptr", p.kv_chunk_size_ptr_offset, int_ws_bytes);
      check_offset("int.merge_indptr", p.merge_indptr_offset, int_ws_bytes);
      check_offset(
          "int.total_num_rows_offset", p.total_num_rows_offset, int_ws_bytes);
      check_offset(
          "int.block_valid_mask", p.block_valid_mask_offset, int_ws_bytes);
      check_offset("float.v_offset", p.v_offset, float_ws_bytes);
      check_offset("float.s_offset", p.s_offset, float_ws_bytes);

      // Dump a few int32 arrays from int workspace to verify scheduler outputs.
      // This helps detect malformed request_indices/qo_tile_indices/o_indptr
      // etc.
      try {
        auto int_ws_cpu = int_workspace_buffer.to(torch::kCPU);
        const uint8_t* base = int_ws_cpu.data_ptr<uint8_t>();
        auto dump_i32 = [&](const char* name, int64_t off_bytes, int64_t n) {
          if (off_bytes < 0 || off_bytes + 4 > int_ws_bytes) return;
          const int32_t* ptr =
              reinterpret_cast<const int32_t*>(base + off_bytes);
          std::ostringstream s;
          s << name << "(off=" << off_bytes << ", n=" << n << "): [";
          int64_t nn = std::min<int64_t>(n, 8);
          for (int64_t i = 0; i < nn; ++i) {
            s << ptr[i] << (i + 1 == nn ? "" : ", ");
          }
          if (n > nn) s << ", ...";
          s << "]";
          LOG(INFO) << "[flashinfer_prefill_debug] " << s.str();
        };
        const int64_t padded_bs = p.padded_batch_size;
        dump_i32("request_indices", p.request_indices_offset, padded_bs);
        dump_i32("qo_tile_indices", p.qo_tile_indices_offset, padded_bs);
        dump_i32("kv_tile_indices", p.kv_tile_indices_offset, padded_bs);
        // o_indptr has length (batch_size+1) but scheduler may pad; print a
        // few.
        dump_i32("o_indptr", p.o_indptr_offset, padded_bs + 1);
        dump_i32("kv_chunk_size_ptr", p.kv_chunk_size_ptr_offset, 1);

        // Also dump the page-locked int buffer head to ensure it's allocated.
        if (page_locked_int_workspace_buffer.defined()) {
          auto pinned_cpu = page_locked_int_workspace_buffer.to(torch::kCPU);
          LOG(INFO) << "[flashinfer_prefill_debug] pinned_int_ws bytes="
                    << pinned_cpu.numel() * pinned_cpu.element_size();
        }
      } catch (const std::exception& e) {
        LOG(WARNING)
            << "[flashinfer_prefill_debug] failed to dump int workspace: "
            << e.what();
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "[flashinfer_prefill_debug] failed to decode plan_info: "
                 << e.what();
  }

  // Indptr sanity
  try {
    auto q_indptr_cpu = q_cu_seq_lens.to(torch::kCPU);
    auto kv_indptr_cpu = kv_cu_seq_lens.to(torch::kCPU);
    int64_t q_last = q_indptr_cpu[-1].item<int64_t>();
    int64_t kv_last = kv_indptr_cpu[-1].item<int64_t>();
    LOG(INFO) << "[flashinfer_prefill_debug] q_indptr numel="
              << q_indptr_cpu.numel() << " last=" << q_last;
    LOG(INFO) << "[flashinfer_prefill_debug] kv_indptr numel="
              << kv_indptr_cpu.numel() << " last=" << kv_last;
    // Print a few values
    std::ostringstream qoss, kvoss;
    qoss << "q_indptr head: [";
    kvoss << "kv_indptr head: [";
    int64_t nprint = std::min<int64_t>(q_indptr_cpu.numel(), 6);
    for (int64_t i = 0; i < nprint; ++i) {
      qoss << q_indptr_cpu[i].item<int64_t>() << (i + 1 == nprint ? "" : ", ");
    }
    qoss << "]";
    nprint = std::min<int64_t>(kv_indptr_cpu.numel(), 6);
    for (int64_t i = 0; i < nprint; ++i) {
      kvoss << kv_indptr_cpu[i].item<int64_t>()
            << (i + 1 == nprint ? "" : ", ");
    }
    kvoss << "]";
    LOG(INFO) << "[flashinfer_prefill_debug] " << qoss.str();
    LOG(INFO) << "[flashinfer_prefill_debug] " << kvoss.str();

    if (is_causal && kv_last < q_last) {
      LOG(ERROR) << "[flashinfer_prefill_debug] causal prefill with kv_last("
                 << kv_last << ") < q_last(" << q_last
                 << ") risks flashinfer underflow.";
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "[flashinfer_prefill_debug] failed to log indptr: "
                 << e.what();
  }
}

void debug_dump_pinned_schedule(const std::string& uri,
                                const torch::Tensor& plan_tensor,
                                const torch::Tensor& pinned_int_ws,
                                const torch::Tensor& qo_indptr_host,
                                const torch::Tensor& kv_indptr_host) {
  try {
    auto plan_cpu = plan_tensor.to(torch::kCPU);
    std::vector<int64_t> vec;
    vec.reserve(plan_cpu.numel());
    for (int64_t i = 0; i < plan_cpu.numel(); ++i)
      vec.push_back(plan_cpu[i].item<int64_t>());

    // Debug: print plan_info_vec size and content before parsing
    std::ostringstream oss;
    oss << "[flashinfer_plan_schedule_debug] " << uri
        << " plan_info_vec(size=" << vec.size() << "): [";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i] << (i + 1 == vec.size() ? "" : ", ");
    }
    oss << "]";
    LOG(INFO) << oss.str();

    auto parsed = parse_plan_info_vec(vec);
    if (!parsed.has_value()) {
      LOG(WARNING) << "[flashinfer_plan_schedule_debug] " << uri
                   << " failed to parse plan_info_vec: expected size=9 (fa2) "
                      "or 15 (fa3), actual size="
                   << vec.size();
      return;
    }
    const auto& p = parsed.value();
    const bool is_fa2 = (vec.size() == 9);
    const int64_t pinned_bytes =
        pinned_int_ws.numel() * pinned_int_ws.element_size();
    const uint8_t* base = pinned_int_ws.data_ptr<uint8_t>();
    auto dump_i32 = [&](const char* name, int64_t off_bytes, int64_t n) {
      if (off_bytes < 0 || off_bytes + 4 > pinned_bytes) return;
      const int32_t* ptr = reinterpret_cast<const int32_t*>(base + off_bytes);
      std::ostringstream s;
      s << name << "(off=" << off_bytes << ", n=" << n << "): [";
      int64_t nn = std::min<int64_t>(n, 8);
      for (int64_t i = 0; i < nn; ++i) {
        s << ptr[i] << (i + 1 == nn ? "" : ", ");
      }
      if (n > nn) s << ", ...";
      s << "]";
      LOG(INFO) << "[flashinfer_plan_schedule_debug] " << uri << " " << s.str();
    };

    LOG(INFO) << "[flashinfer_plan_schedule_debug] " << uri
              << " format=" << (is_fa2 ? "fa2" : "fa3")
              << " pinned_int_ws bytes=" << pinned_bytes
              << " padded_batch_size=" << p.padded_batch_size
              << " cta_tile_q=" << p.cta_tile_q;
    if (!is_fa2) {
      LOG(INFO) << "[flashinfer_plan_schedule_debug] " << uri
                << " split_kv=" << p.split_kv;
    }

    // Also print indptrs (host) for context
    LOG(INFO) << "[flashinfer_plan_schedule_debug] qo_indptr_host="
              << qo_indptr_host;
    LOG(INFO) << "[flashinfer_plan_schedule_debug] kv_indptr_host="
              << kv_indptr_host;

    dump_i32("request_indices", p.request_indices_offset, p.padded_batch_size);
    dump_i32("qo_tile_indices", p.qo_tile_indices_offset, p.padded_batch_size);
    dump_i32("kv_tile_indices", p.kv_tile_indices_offset, p.padded_batch_size);
    dump_i32("o_indptr", p.o_indptr_offset, (qo_indptr_host.numel()));
    // kv_chunk_size_ptr is only available in fa3 format
    if (!is_fa2) {
      dump_i32("kv_chunk_size_ptr", p.kv_chunk_size_ptr_offset, 1);
    }
  } catch (const std::exception& e) {
    LOG(WARNING)
        << "[flashinfer_plan_schedule_debug] failed to dump pinned schedule: "
        << e.what();
  }
}

}  // namespace xllm::kernel::cuda
