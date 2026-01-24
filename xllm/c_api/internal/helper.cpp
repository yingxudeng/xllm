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

#include "helper.h"

#include <glog/logging.h>
#include <pthread.h>
#include <torch/torch.h>

#include <atomic>
#include <string>

#include "core/common/global_flags.h"
#include "core/util/env_var.h"
#include "core/util/uuid.h"

namespace xllm {
namespace helper {
namespace {
thread_local ShortUUID short_uuid;
static std::atomic<bool> g_glog_inited = false;
static pthread_mutex_t g_log_init_mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

std::string generate_request_id() {
  return "xllm-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

void init_log(const std::string& log_dir) {
  if (g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (!g_glog_inited.load(std::memory_order_relaxed)) {
    google::InitGoogleLogging("xllm");

    std::string log_prefix = log_dir.empty() ? "./" : log_dir + "/";
    google::SetLogDestination(google::INFO,
                              (log_prefix + "xllm.log.INFO.").c_str());
    google::SetLogDestination(google::WARNING,
                              (log_prefix + "xllm.log.WARNING.").c_str());
    google::SetLogDestination(google::ERROR,
                              (log_prefix + "xllm.log.ERROR.").c_str());
    google::SetStderrLogging(google::FATAL);
    g_glog_inited.store(true, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void shutdown_log() {
  if (!g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (g_glog_inited.load(std::memory_order_relaxed)) {
    google::ShutdownGoogleLogging();
    g_glog_inited.store(false, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void set_init_options(BackendType backend_type,
                      const XLLM_InitOptions* init_options,
                      XLLM_InitOptions* xllm_init_options) {
  if (NULL == init_options) {
    if (backend_type == BackendType::LLM) {
      memcpy(xllm_init_options,
             &XLLM_INIT_LLM_OPTIONS_DEFAULT,
             sizeof(XLLM_InitOptions));
    } else if (backend_type == BackendType::REC) {
      memcpy(xllm_init_options,
             &XLLM_INIT_REC_OPTIONS_DEFAULT,
             sizeof(XLLM_InitOptions));
    }
  } else {
    memcpy(xllm_init_options, init_options, sizeof(XLLM_InitOptions));
  }

  return;
}

namespace {
// Helper function to override string field from environment variable
void override_string_field(char* field,
                           size_t field_size,
                           const std::string& env_key,
                           const char* default_value) {
  std::string value =
      xllm::util::get_string_env_opt(env_key, std::string(default_value));
  strncpy(field, value.c_str(), field_size - 1);
  field[field_size - 1] = '\0';
}
}  // namespace

void override_init_options_from_env(const std::string& env_prefix,
                                    XLLM_InitOptions* opts) {
  // Boolean options
  opts->enable_mla =
      xllm::util::get_bool_env(env_prefix + "ENABLE_MLA", opts->enable_mla);
  opts->enable_chunked_prefill = xllm::util::get_bool_env(
      env_prefix + "ENABLE_CHUNKED_PREFILL", opts->enable_chunked_prefill);
  opts->enable_prefix_cache = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PREFIX_CACHE", opts->enable_prefix_cache);
  opts->enable_disagg_pd = xllm::util::get_bool_env(
      env_prefix + "ENABLE_DISAGG_PD", opts->enable_disagg_pd);
  opts->enable_pd_ooc = xllm::util::get_bool_env(env_prefix + "ENABLE_PD_OOC",
                                                 opts->enable_pd_ooc);
  opts->enable_schedule_overlap = xllm::util::get_bool_env(
      env_prefix + "ENABLE_SCHEDULE_OVERLAP", opts->enable_schedule_overlap);
  opts->enable_shm =
      xllm::util::get_bool_env(env_prefix + "ENABLE_SHM", opts->enable_shm);

  // Integer options
  opts->transfer_listen_port = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "TRANSFER_LISTEN_PORT", opts->transfer_listen_port));
  opts->nnodes = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "NNODES", opts->nnodes));
  opts->node_rank = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "NODE_RANK", opts->node_rank));
  opts->dp_size = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "DP_SIZE", opts->dp_size));
  opts->ep_size = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "EP_SIZE", opts->ep_size));
  opts->block_size = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "BLOCK_SIZE", opts->block_size));
  opts->max_cache_size = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_CACHE_SIZE", opts->max_cache_size));
  opts->max_tokens_per_batch = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_TOKENS_PER_BATCH", opts->max_tokens_per_batch));
  opts->max_seqs_per_batch = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_SEQS_PER_BATCH", opts->max_seqs_per_batch));
  opts->max_tokens_per_chunk_for_prefill = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "MAX_TOKENS_PER_CHUNK_FOR_PREFILL",
                              opts->max_tokens_per_chunk_for_prefill));
  opts->num_speculative_tokens = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "NUM_SPECULATIVE_TOKENS", opts->num_speculative_tokens));
  opts->num_request_handling_threads = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "NUM_REQUEST_HANDLING_THREADS",
                              opts->num_request_handling_threads));
  opts->expert_parallel_degree = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "EXPERT_PARALLEL_DEGREE", opts->expert_parallel_degree));
  opts->server_idx = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "SERVER_IDX", opts->server_idx));
  opts->beam_width = static_cast<uint32_t>(
      xllm::util::get_int_env(env_prefix + "BEAM_WIDTH", opts->beam_width));
  opts->max_decode_rounds = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_DECODE_ROUNDS", opts->max_decode_rounds));
  opts->max_token_per_req = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_TOKEN_PER_REQ", opts->max_token_per_req));

  // Float options
  opts->max_memory_utilization = static_cast<float>(xllm::util::get_double_env(
      env_prefix + "MAX_MEMORY_UTILIZATION", opts->max_memory_utilization));

  // String options
  override_string_field(
      opts->task, sizeof(opts->task), env_prefix + "TASK", opts->task);
  override_string_field(opts->communication_backend,
                        sizeof(opts->communication_backend),
                        env_prefix + "COMMUNICATION_BACKEND",
                        opts->communication_backend);
  override_string_field(opts->device_ip,
                        sizeof(opts->device_ip),
                        env_prefix + "DEVICE_IP",
                        opts->device_ip);
  override_string_field(opts->master_node_addr,
                        sizeof(opts->master_node_addr),
                        env_prefix + "MASTER_NODE_ADDR",
                        opts->master_node_addr);
  override_string_field(opts->xservice_addr,
                        sizeof(opts->xservice_addr),
                        env_prefix + "XSERVICE_ADDR",
                        opts->xservice_addr);
  override_string_field(opts->instance_name,
                        sizeof(opts->instance_name),
                        env_prefix + "INSTANCE_NAME",
                        opts->instance_name);
  override_string_field(opts->kv_cache_transfer_mode,
                        sizeof(opts->kv_cache_transfer_mode),
                        env_prefix + "KV_CACHE_TRANSFER_MODE",
                        opts->kv_cache_transfer_mode);
  override_string_field(opts->log_dir,
                        sizeof(opts->log_dir),
                        env_prefix + "LOG_DIR",
                        opts->log_dir);
  override_string_field(opts->draft_model,
                        sizeof(opts->draft_model),
                        env_prefix + "DRAFT_MODEL",
                        opts->draft_model);
  override_string_field(opts->draft_devices,
                        sizeof(opts->draft_devices),
                        env_prefix + "DRAFT_DEVICES",
                        opts->draft_devices);
}

void override_global_flags_from_env(const std::string& env_prefix,
                                    BackendType backend_type) {
  // Service config
  FLAGS_host = xllm::util::get_string_env_opt(env_prefix + "HOST", FLAGS_host);
  FLAGS_port = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "PORT", FLAGS_port));
  FLAGS_rpc_idle_timeout_s = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "RPC_IDLE_TIMEOUT_S", FLAGS_rpc_idle_timeout_s));
  FLAGS_rpc_channel_timeout_ms = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "RPC_CHANNEL_TIMEOUT_MS", FLAGS_rpc_channel_timeout_ms));
  FLAGS_max_reconnect_count = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_RECONNECT_COUNT", FLAGS_max_reconnect_count));
  FLAGS_num_threads = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NUM_THREADS", FLAGS_num_threads));
  FLAGS_max_concurrent_requests = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_CONCURRENT_REQUESTS", FLAGS_max_concurrent_requests));

  // Model config
  FLAGS_model_id =
      xllm::util::get_string_env_opt(env_prefix + "MODEL_ID", FLAGS_model_id);
  FLAGS_model =
      xllm::util::get_string_env_opt(env_prefix + "MODEL", FLAGS_model);
  FLAGS_backend =
      xllm::util::get_string_env_opt(env_prefix + "BACKEND", FLAGS_backend);
  FLAGS_task =
      xllm::util::get_string_env_opt(env_prefix + "TASK_FLAG", FLAGS_task);
  FLAGS_devices =
      xllm::util::get_string_env_opt(env_prefix + "DEVICES", FLAGS_devices);
  FLAGS_draft_model = xllm::util::get_string_env_opt(
      env_prefix + "DRAFT_MODEL_FLAG", FLAGS_draft_model);
  FLAGS_draft_devices = xllm::util::get_string_env_opt(
      env_prefix + "DRAFT_DEVICES_FLAG", FLAGS_draft_devices);
  FLAGS_enable_mla = xllm::util::get_bool_env(env_prefix + "ENABLE_MLA_FLAG",
                                              FLAGS_enable_mla);
  FLAGS_enable_customize_mla_kernel =
      xllm::util::get_bool_env(env_prefix + "ENABLE_CUSTOMIZE_MLA_KERNEL",
                               FLAGS_enable_customize_mla_kernel);

  // Graph mode execution config
  FLAGS_max_seq_len_for_graph_mode = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "MAX_SEQ_LEN_FOR_GRAPH_MODE",
                              FLAGS_max_seq_len_for_graph_mode));

  // For REC backend, default enable graph modes
  bool graph_default = (backend_type == BackendType::REC);
  FLAGS_enable_graph =
      xllm::util::get_bool_env(env_prefix + "ENABLE_GRAPH", graph_default);
  FLAGS_enable_graph_no_padding = xllm::util::get_bool_env(
      env_prefix + "ENABLE_GRAPH_NO_PADDING", graph_default);
  FLAGS_enable_prefill_piecewise_graph = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PREFILL_PIECEWISE_GRAPH", graph_default);

  // VLM config
  FLAGS_limit_image_per_prompt = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "LIMIT_IMAGE_PER_PROMPT", FLAGS_limit_image_per_prompt));

  // Threading config
  FLAGS_num_request_handling_threads = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NUM_REQUEST_HANDLING_THREADS_FLAG",
                              FLAGS_num_request_handling_threads));
  FLAGS_num_response_handling_threads = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NUM_RESPONSE_HANDLING_THREADS",
                              FLAGS_num_response_handling_threads));

  // KVCache config
  FLAGS_block_size = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "BLOCK_SIZE_FLAG", FLAGS_block_size));
  FLAGS_max_cache_size = xllm::util::get_int_env(
      env_prefix + "MAX_CACHE_SIZE_FLAG", FLAGS_max_cache_size);
  FLAGS_max_memory_utilization = xllm::util::get_double_env(
      env_prefix + "MAX_MEMORY_UTILIZATION_FLAG", FLAGS_max_memory_utilization);

  // Scheduler config
  FLAGS_max_tokens_per_batch = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_TOKENS_PER_BATCH_FLAG", FLAGS_max_tokens_per_batch));
  FLAGS_enable_schedule_overlap =
      xllm::util::get_bool_env(env_prefix + "ENABLE_SCHEDULE_OVERLAP_FLAG",
                               FLAGS_enable_schedule_overlap);
  FLAGS_prefill_scheduling_memory_usage_threshold = xllm::util::get_double_env(
      env_prefix + "PREFILL_SCHEDULING_MEMORY_USAGE_THRESHOLD",
      FLAGS_prefill_scheduling_memory_usage_threshold);
  FLAGS_enable_chunked_prefill = xllm::util::get_bool_env(
      env_prefix + "ENABLE_CHUNKED_PREFILL_FLAG", FLAGS_enable_chunked_prefill);
  FLAGS_max_tokens_per_chunk_for_prefill =
      static_cast<int32_t>(xllm::util::get_int_env(
          env_prefix + "MAX_TOKENS_PER_CHUNK_FOR_PREFILL_FLAG",
          FLAGS_max_tokens_per_chunk_for_prefill));
  FLAGS_chunked_match_frequency = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "CHUNKED_MATCH_FREQUENCY", FLAGS_chunked_match_frequency));
  FLAGS_use_zero_evict = xllm::util::get_bool_env(env_prefix + "USE_ZERO_EVICT",
                                                  FLAGS_use_zero_evict);
  FLAGS_max_decode_token_per_sequence = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "MAX_DECODE_TOKEN_PER_SEQUENCE",
                              FLAGS_max_decode_token_per_sequence));
  FLAGS_request_queue_size = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "REQUEST_QUEUE_SIZE", FLAGS_request_queue_size));

  // Parallel config
  FLAGS_dp_size = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "DP_SIZE_FLAG", FLAGS_dp_size));
  FLAGS_ep_size = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "EP_SIZE_FLAG", FLAGS_ep_size));
  FLAGS_communication_backend = xllm::util::get_string_env_opt(
      env_prefix + "COMMUNICATION_BACKEND_FLAG", FLAGS_communication_backend);

  // EP load balance config
  FLAGS_enable_eplb =
      xllm::util::get_bool_env(env_prefix + "ENABLE_EPLB", FLAGS_enable_eplb);
  FLAGS_redundant_experts_num = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "REDUNDANT_EXPERTS_NUM", FLAGS_redundant_experts_num));
  FLAGS_eplb_update_interval = xllm::util::get_int_env(
      env_prefix + "EPLB_UPDATE_INTERVAL", FLAGS_eplb_update_interval);
  FLAGS_eplb_update_threshold = xllm::util::get_double_env(
      env_prefix + "EPLB_UPDATE_THRESHOLD", FLAGS_eplb_update_threshold);
  FLAGS_expert_parallel_degree = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "EXPERT_PARALLEL_DEGREE_FLAG",
                              FLAGS_expert_parallel_degree));
  FLAGS_rank_tablefile = xllm::util::get_string_env_opt(
      env_prefix + "RANK_TABLEFILE", FLAGS_rank_tablefile);

  // Profile config
  FLAGS_enable_profile_step_time = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PROFILE_STEP_TIME", FLAGS_enable_profile_step_time);
  FLAGS_enable_profile_token_budget =
      xllm::util::get_bool_env(env_prefix + "ENABLE_PROFILE_TOKEN_BUDGET",
                               FLAGS_enable_profile_token_budget);
  FLAGS_enable_latency_aware_schedule =
      xllm::util::get_bool_env(env_prefix + "ENABLE_LATENCY_AWARE_SCHEDULE",
                               FLAGS_enable_latency_aware_schedule);
  FLAGS_profile_max_prompt_length = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "PROFILE_MAX_PROMPT_LENGTH",
                              FLAGS_profile_max_prompt_length));
  FLAGS_enable_profile_kv_blocks = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PROFILE_KV_BLOCKS", FLAGS_enable_profile_kv_blocks);
  FLAGS_disable_ttft_profiling = xllm::util::get_bool_env(
      env_prefix + "DISABLE_TTFT_PROFILING", FLAGS_disable_ttft_profiling);
  FLAGS_enable_forward_interruption =
      xllm::util::get_bool_env(env_prefix + "ENABLE_FORWARD_INTERRUPTION",
                               FLAGS_enable_forward_interruption);
  FLAGS_max_global_ttft_ms = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_GLOBAL_TTFT_MS", FLAGS_max_global_ttft_ms));
  FLAGS_max_global_tpot_ms = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_GLOBAL_TPOT_MS", FLAGS_max_global_tpot_ms));

  // Prefix cache config
  FLAGS_enable_prefix_cache = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PREFIX_CACHE_FLAG", FLAGS_enable_prefix_cache);
  FLAGS_enable_cache_upload = xllm::util::get_bool_env(
      env_prefix + "ENABLE_CACHE_UPLOAD", FLAGS_enable_cache_upload);
  FLAGS_murmur_hash3_seed = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "MURMUR_HASH3_SEED", FLAGS_murmur_hash3_seed));

  // Multi-nodes config
  FLAGS_master_node_addr = xllm::util::get_string_env_opt(
      env_prefix + "MASTER_NODE_ADDR_FLAG", FLAGS_master_node_addr);
  FLAGS_nnodes = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NNODES_FLAG", FLAGS_nnodes));
  FLAGS_node_rank = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NODE_RANK_FLAG", FLAGS_node_rank));
  FLAGS_enable_shm = xllm::util::get_bool_env(env_prefix + "ENABLE_SHM_FLAG",
                                              FLAGS_enable_shm);
  FLAGS_use_contiguous_input_buffer =
      xllm::util::get_bool_env(env_prefix + "USE_CONTIGUOUS_INPUT_BUFFER",
                               FLAGS_use_contiguous_input_buffer);

  // Disaggregated PD config
  FLAGS_xservice_addr = xllm::util::get_string_env_opt(
      env_prefix + "XSERVICE_ADDR_FLAG", FLAGS_xservice_addr);
  FLAGS_enable_disagg_pd = xllm::util::get_bool_env(
      env_prefix + "ENABLE_DISAGG_PD_FLAG", FLAGS_enable_disagg_pd);
  FLAGS_enable_pd_ooc = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PD_OOC_FLAG", FLAGS_enable_pd_ooc);
  FLAGS_disagg_pd_port = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "DISAGG_PD_PORT", FLAGS_disagg_pd_port));
  FLAGS_instance_role = xllm::util::get_string_env_opt(
      env_prefix + "INSTANCE_ROLE", FLAGS_instance_role);
  FLAGS_kv_cache_transfer_type = xllm::util::get_string_env_opt(
      env_prefix + "KV_CACHE_TRANSFER_TYPE", FLAGS_kv_cache_transfer_type);
  FLAGS_kv_cache_transfer_mode = xllm::util::get_string_env_opt(
      env_prefix + "KV_CACHE_TRANSFER_MODE_FLAG", FLAGS_kv_cache_transfer_mode);
  FLAGS_npu_phy_id = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NPU_PHY_ID", FLAGS_npu_phy_id));
  FLAGS_transfer_listen_port = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "TRANSFER_LISTEN_PORT_FLAG", FLAGS_transfer_listen_port));

  // Function call config
  FLAGS_tool_call_parser = xllm::util::get_string_env_opt(
      env_prefix + "TOOL_CALL_PARSER", FLAGS_tool_call_parser);

  // Speculative config
  FLAGS_num_speculative_tokens = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "NUM_SPECULATIVE_TOKENS_FLAG",
                              FLAGS_num_speculative_tokens));
  FLAGS_enable_atb_spec_kernel = xllm::util::get_bool_env(
      env_prefix + "ENABLE_ATB_SPEC_KERNEL", FLAGS_enable_atb_spec_kernel);

  // Block copy config
  FLAGS_enable_block_copy_kernel = xllm::util::get_bool_env(
      env_prefix + "ENABLE_BLOCK_COPY_KERNEL", FLAGS_enable_block_copy_kernel);

  // Service routing config
  FLAGS_etcd_addr =
      xllm::util::get_string_env_opt(env_prefix + "ETCD_ADDR", FLAGS_etcd_addr);
  FLAGS_enable_service_routing = xllm::util::get_bool_env(
      env_prefix + "ENABLE_SERVICE_ROUTING", FLAGS_enable_service_routing);
  FLAGS_heart_beat_interval = xllm::util::get_double_env(
      env_prefix + "HEART_BEAT_INTERVAL", FLAGS_heart_beat_interval);
  FLAGS_etcd_ttl = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "ETCD_TTL", FLAGS_etcd_ttl));

  // Priority strategy config
  FLAGS_priority_strategy = xllm::util::get_string_env_opt(
      env_prefix + "PRIORITY_STRATEGY", FLAGS_priority_strategy);
  FLAGS_enable_online_preempt_offline =
      xllm::util::get_bool_env(env_prefix + "ENABLE_ONLINE_PREEMPT_OFFLINE",
                               FLAGS_enable_online_preempt_offline);

  // KVCache store config
  FLAGS_prefetch_timeout = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "PREFETCH_TIMEOUT", FLAGS_prefetch_timeout));
  FLAGS_prefetch_bacth_size = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "PREFETCH_BATCH_SIZE", FLAGS_prefetch_bacth_size));
  FLAGS_layers_wise_copy_batchs = static_cast<uint32_t>(xllm::util::get_int_env(
      env_prefix + "LAYERS_WISE_COPY_BATCHS", FLAGS_layers_wise_copy_batchs));
  FLAGS_host_blocks_factor = xllm::util::get_double_env(
      env_prefix + "HOST_BLOCKS_FACTOR", FLAGS_host_blocks_factor);
  FLAGS_enable_kvcache_store = xllm::util::get_bool_env(
      env_prefix + "ENABLE_KVCACHE_STORE", FLAGS_enable_kvcache_store);
  FLAGS_store_protocol = xllm::util::get_string_env_opt(
      env_prefix + "STORE_PROTOCOL", FLAGS_store_protocol);
  FLAGS_store_master_server_address =
      xllm::util::get_string_env_opt(env_prefix + "STORE_MASTER_SERVER_ADDRESS",
                                     FLAGS_store_master_server_address);
  FLAGS_store_metadata_server = xllm::util::get_string_env_opt(
      env_prefix + "STORE_METADATA_SERVER", FLAGS_store_metadata_server);
  FLAGS_store_local_hostname = xllm::util::get_string_env_opt(
      env_prefix + "STORE_LOCAL_HOSTNAME", FLAGS_store_local_hostname);

  // Computation communication parallel config
  FLAGS_enable_multi_stream_parallel =
      xllm::util::get_bool_env(env_prefix + "ENABLE_MULTI_STREAM_PARALLEL",
                               FLAGS_enable_multi_stream_parallel);
  FLAGS_micro_batch_num = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MICRO_BATCH_NUM", FLAGS_micro_batch_num));

  // DIT config
  FLAGS_max_requests_per_batch = static_cast<int32_t>(xllm::util::get_int_env(
      env_prefix + "MAX_REQUESTS_PER_BATCH", FLAGS_max_requests_per_batch));

  // Continuous KVCache config
  FLAGS_enable_continuous_kvcache =
      xllm::util::get_bool_env(env_prefix + "ENABLE_CONTINUOUS_KVCACHE",
                               FLAGS_enable_continuous_kvcache);
  FLAGS_phy_page_granularity_size =
      xllm::util::get_int_env(env_prefix + "PHY_PAGE_GRANULARITY_SIZE",
                              FLAGS_phy_page_granularity_size);
  FLAGS_cache_size_per_token = xllm::util::get_int_env(
      env_prefix + "CACHE_SIZE_PER_TOKEN", FLAGS_cache_size_per_token);
  FLAGS_buffer_size_per_seq = xllm::util::get_int_env(
      env_prefix + "BUFFER_SIZE_PER_SEQ", FLAGS_buffer_size_per_seq);

  // Beam search config
  FLAGS_enable_beam_search_kernel =
      xllm::util::get_bool_env(env_prefix + "ENABLE_BEAM_SEARCH_KERNEL",
                               FLAGS_enable_beam_search_kernel);

  // Reasoning parser config
  FLAGS_reasoning_parser = xllm::util::get_string_env_opt(
      env_prefix + "REASONING_PARSER", FLAGS_reasoning_parser);

  // Qwen3 reranker config
  FLAGS_enable_qwen3_reranker = xllm::util::get_bool_env(
      env_prefix + "ENABLE_QWEN3_RERANKER", FLAGS_enable_qwen3_reranker);

  // Flashinfer config
  FLAGS_flashinfer_workspace_buffer_size = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "FLASHINFER_WORKSPACE_BUFFER_SIZE",
                              FLAGS_flashinfer_workspace_buffer_size));

  // Prefetch weight config
  FLAGS_enable_prefetch_weight = xllm::util::get_bool_env(
      env_prefix + "ENABLE_PREFETCH_WEIGHT", FLAGS_enable_prefetch_weight);

  // DP load balance
  FLAGS_enable_dp_balance = xllm::util::get_bool_env(
      env_prefix + "ENABLE_DP_BALANCE", FLAGS_enable_dp_balance);

  // Random seed
  FLAGS_random_seed = static_cast<int32_t>(
      xllm::util::get_int_env(env_prefix + "RANDOM_SEED", FLAGS_random_seed));

  // DIT cache config
  FLAGS_dit_cache_policy = xllm::util::get_string_env_opt(
      env_prefix + "DIT_CACHE_POLICY", FLAGS_dit_cache_policy);
  FLAGS_dit_cache_warmup_steps = xllm::util::get_int_env(
      env_prefix + "DIT_CACHE_WARMUP_STEPS", FLAGS_dit_cache_warmup_steps);
  FLAGS_dit_cache_n_derivatives = xllm::util::get_int_env(
      env_prefix + "DIT_CACHE_N_DERIVATIVES", FLAGS_dit_cache_n_derivatives);
  FLAGS_dit_cache_skip_interval_steps =
      xllm::util::get_int_env(env_prefix + "DIT_CACHE_SKIP_INTERVAL_STEPS",
                              FLAGS_dit_cache_skip_interval_steps);
  FLAGS_dit_cache_residual_diff_threshold = xllm::util::get_double_env(
      env_prefix + "DIT_CACHE_RESIDUAL_DIFF_THRESHOLD",
      FLAGS_dit_cache_residual_diff_threshold);

  // Constrained decoding
  FLAGS_enable_constrained_decoding =
      xllm::util::get_bool_env(env_prefix + "ENABLE_CONSTRAINED_DECODING",
                               FLAGS_enable_constrained_decoding);

  // Sampler config
  FLAGS_enable_fast_sampler = xllm::util::get_bool_env(
      env_prefix + "ENABLE_FAST_SAMPLER", FLAGS_enable_fast_sampler);
  FLAGS_enable_topk_sorted = xllm::util::get_bool_env(
      env_prefix + "ENABLE_TOPK_SORTED", FLAGS_enable_topk_sorted);

  // Concurrent rec worker config (REC specific, default 2 for REC)
  uint32_t rec_worker_default = (backend_type == BackendType::REC) ? 2 : 1;
  FLAGS_rec_worker_max_concurrency =
      static_cast<uint32_t>(xllm::util::get_int_env(
          env_prefix + "REC_WORKER_MAX_CONCURRENCY", rec_worker_default));

  // XAttention config
  FLAGS_enable_xattention_two_stage_decode = xllm::util::get_bool_env(
      env_prefix + "ENABLE_XATTENTION_TWO_STAGE_DECODE",
      FLAGS_enable_xattention_two_stage_decode);

  // Qwen3 fused qk norm rope config
  FLAGS_enable_qwen3_fused_qk_norm_rope_kernel = xllm::util::get_bool_env(
      env_prefix + "ENABLE_QWEN3_FUSED_QK_NORM_ROPE_KERNEL",
      FLAGS_enable_qwen3_fused_qk_norm_rope_kernel);
}

void transfer_request_params(InferenceType inference_type,
                             const XLLM_RequestParams* request_params,
                             xllm::RequestParams* xllm_request_params) {
  XLLM_RequestParams final_request_params;
  if (nullptr == request_params) {
    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::LLM_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &XLLM_LLM_REQUEST_PARAMS_DEFAULT,
             sizeof(XLLM_RequestParams));
    } else if (inference_type == InferenceType::REC_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &XLLM_REC_REQUEST_PARAMS_DEFAULT,
             sizeof(XLLM_RequestParams));
    }
  } else {
    memcpy(&final_request_params, request_params, sizeof(XLLM_RequestParams));
  }

  xllm_request_params->echo = final_request_params.echo;
  xllm_request_params->offline = final_request_params.offline;
  xllm_request_params->logprobs = final_request_params.logprobs;
  xllm_request_params->ignore_eos = final_request_params.ignore_eos;

  xllm_request_params->best_of = final_request_params.best_of;
  xllm_request_params->slo_ms = final_request_params.slo_ms;
  xllm_request_params->top_k = final_request_params.top_k;
  xllm_request_params->top_p = final_request_params.top_p;
  xllm_request_params->n = final_request_params.n;
  xllm_request_params->max_tokens = final_request_params.max_tokens;
  xllm_request_params->frequency_penalty =
      final_request_params.frequency_penalty;
  xllm_request_params->presence_penalty = final_request_params.presence_penalty;
  xllm_request_params->repetition_penalty =
      final_request_params.repetition_penalty;
  xllm_request_params->beam_width = final_request_params.beam_width;
  xllm_request_params->top_logprobs = final_request_params.top_logprobs;
  xllm_request_params->temperature = final_request_params.temperature;
  xllm_request_params->request_id = final_request_params.request_id;

  return;
}

XLLM_Response* build_error_response(const std::string& request_id,
                                    XLLM_StatusCode status_code,
                                    const std::string& error_info) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = status_code;
  strncpy(
      response->error_info, error_info.c_str(), XLLM_ERROR_INFO_MAX_LEN - 1);
  response->error_info[XLLM_ERROR_INFO_MAX_LEN - 1] = '\0';

  XLLM_SET_META_STRING_FIELD(response->id, request_id);

  LOG(ERROR) << "Request [" << request_id << "] error: " << error_info
             << " (code: " << static_cast<int>(response->status_code) << ")";

  return response;
}

XLLM_Response* build_success_response(const InferenceType& inference_type,
                                      const RequestOutput& output,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = XLLM_StatusCode::kSuccess;
  response->created = created_time;
  XLLM_SET_META_STRING_FIELD(response->id, request_id);
  XLLM_SET_META_STRING_FIELD(response->model, model);

  if (inference_type == InferenceType::LLM_COMPLETIONS ||
      inference_type == InferenceType::REC_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "text_completion");
  } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
             inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "chat.completion");
  }

  response->choices.entries_size = output.outputs.size();
  response->choices.entries = new XLLM_Choice[response->choices.entries_size]();
  CHECK(nullptr != response->choices.entries);

  for (int i = 0; i < output.outputs.size(); i++) {
    const auto& seq_output = output.outputs[i];
    XLLM_Choice& choice = response->choices.entries[i];
    choice.index = seq_output.index;

    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::REC_COMPLETIONS) {
      size_t text_len = seq_output.text.length();
      choice.text = new char[text_len + 1];
      CHECK(nullptr != choice.text);
      strncpy(choice.text, seq_output.text.c_str(), text_len + 1);
      choice.text[text_len] = '\0';
    } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      choice.message = new XLLM_ChatMessage();
      CHECK(nullptr != choice.message);

      snprintf(choice.message->role, sizeof(choice.message->role), "assistant");
      size_t text_len = seq_output.text.length();
      choice.message->content = new char[text_len + 1];
      CHECK(nullptr != choice.message->content);
      strncpy(choice.message->content, seq_output.text.c_str(), text_len + 1);
      choice.message->content[text_len] = '\0';
    }

    if (seq_output.finish_reason.has_value()) {
      XLLM_SET_META_STRING_FIELD(choice.finish_reason,
                                 seq_output.finish_reason.value());
    }

    if (seq_output.token_ids.size() > 0) {
      choice.token_size = seq_output.token_ids.size();
      choice.token_ids = new int32_t[choice.token_size];
      CHECK(nullptr != choice.token_ids);
      for (int j = 0; j < choice.token_size; j++) {
        choice.token_ids[j] = seq_output.token_ids[j];
      }
    }

    if (seq_output.logprobs.has_value()) {
      choice.logprobs.entries_size = seq_output.logprobs.value().size();
      choice.logprobs.entries =
          new XLLM_LogProb[choice.logprobs.entries_size]();
      CHECK(nullptr != choice.logprobs.entries);
      for (int j = 0; j < seq_output.logprobs.value().size(); j++) {
        const auto& logprob = seq_output.logprobs.value()[j];
        XLLM_LogProb& xllm_logprob = choice.logprobs.entries[j];

        xllm_logprob.token_id = logprob.token_id;
        xllm_logprob.logprob = logprob.logprob;
      }
    }
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response->usage.prompt_tokens =
        static_cast<int32_t>(usage.num_prompt_tokens);
    response->usage.completion_tokens =
        static_cast<int32_t>(usage.num_generated_tokens);
    response->usage.total_tokens = static_cast<int32_t>(usage.num_total_tokens);
  }

  return response;
}

template <typename HandlerType, typename InputType>
XLLM_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  CHECK(nullptr != handler);

  std::string request_id;
  if (nullptr != request_params && strlen(request_params->request_id) > 0) {
    request_id = request_params->request_id;
  } else {
    request_id = generate_request_id();
  }

  if (!handler->initialized) {
    return build_error_response(
        request_id, XLLM_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  if (std::find(handler->model_ids.begin(),
                handler->model_ids.end(),
                model_id) == handler->model_ids.end()) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  xllm::RequestParams xllm_request_params;
  transfer_request_params(inference_type, request_params, &xllm_request_params);
  xllm_request_params.request_id = request_id;

  const int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<XLLM_Response*>>();
    auto future = promise_ptr->getSemiFuture();

    auto on_request_complete = [model_id,
                                request_id,
                                created_time,
                                inference_type,
                                weak_promise = std::weak_ptr(promise_ptr)](
                                   const RequestOutput& req_output) -> bool {
      if (auto locked_promise = weak_promise.lock()) {
        try {
          locked_promise->setValue(build_success_response(
              inference_type, req_output, request_id, created_time, model_id));
          return true;
        } catch (const std::exception& e) {
          LOG(ERROR) << "Build response failed: " << e.what();
          locked_promise->setValue(build_error_response(
              request_id,
              XLLM_StatusCode::kInternalError,
              "Build response failed: " + std::string(e.what())));
        }
      }
      return false;
    };

    if constexpr (std::is_same_v<HandlerType, XLLM_LLM_Handler>) {
      handler->master->handle_request(input,
                                      std::nullopt,
                                      xllm_request_params,
                                      std::nullopt,
                                      on_request_complete);
    } else if constexpr (std::is_same_v<HandlerType, XLLM_REC_Handler>) {
      if constexpr (std::is_same_v<InputType, std::vector<int>>) {
        if (nullptr != extra) {
          xllm::MMData* mm_data =
              dynamic_cast<xllm::MMData*>(static_cast<xllm::MMData*>(extra));
          CHECK(nullptr != mm_data);

          std::optional<xllm::MMData> opt_mm_data = std::move(*mm_data);
          handler->master->handle_request(
              input, opt_mm_data, xllm_request_params, on_request_complete);

        } else {
          handler->master->handle_request("",
                                          input,
                                          std::nullopt,
                                          xllm_request_params,
                                          on_request_complete);
        }
      } else {
        handler->master->handle_request(input,
                                        std::nullopt,
                                        std::nullopt,
                                        xllm_request_params,
                                        on_request_complete);
      }
    } else {
      CHECK(false);
    }

    return std::move(future)
        .via(handler->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([request_id](
                     folly::Try<XLLM_Response*>&& result) -> XLLM_Response* {
          if (result.hasValue()) return std::move(result).value();

          std::string error_msg;
          XLLM_StatusCode code = XLLM_StatusCode::kInternalError;
          try {
            result.throwUnlessValue();
          } catch (const folly::FutureTimeout& e) {
            error_msg = "Request timed out: " + std::string(e.what());
            code = XLLM_StatusCode::kTimeout;
          } catch (const std::exception& e) {
            error_msg = "Inference failed: " + std::string(e.what());
          } catch (...) {
            error_msg = "Inference failed with unknown exception";
          }
          return build_error_response(request_id, code, error_msg);
        })
        .get();

  } catch (...) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kInternalError,
                                "Critical error in inference pipeline");
  }
}

void xllm_free_response(XLLM_Response* resp) {
  if (nullptr == resp) {
    return;
  }

  if (nullptr != resp->choices.entries) {
    for (int i = 0; i < resp->choices.entries_size; ++i) {
      XLLM_Choice& choice = resp->choices.entries[i];

      if (nullptr != choice.text) {
        delete[] choice.text;
        choice.text = nullptr;
      }

      if (nullptr != choice.message) {
        if (nullptr != choice.message->content) {
          delete[] choice.message->content;
          choice.message->content = nullptr;
        }
        delete choice.message;
        choice.message = nullptr;
      }

      if (nullptr != choice.token_ids) {
        delete[] choice.token_ids;
        choice.token_ids = nullptr;
        choice.token_size = 0;
      }

      if (nullptr != choice.logprobs.entries) {
        delete[] choice.logprobs.entries;
        choice.logprobs.entries = nullptr;
      }
      choice.logprobs.entries_size = 0;
    }

    delete[] resp->choices.entries;
    resp->choices.entries = nullptr;
  }

  resp->choices.entries_size = 0;
  delete resp;

  return;
}

torch::ScalarType xllm_dtype_to_torch_scalar_type(XLLM_DataType dtype) {
  switch (dtype) {
    case XLLM_DTYPE_UNDEFINED:
      throw std::runtime_error(
          "XLLM_DTYPE_UNDEFINED is not a valid dtype for tensor conversion");
    case XLLM_DTYPE_FLOAT16:
      return torch::kFloat16;
    case XLLM_DTYPE_FLOAT32:
      return torch::kFloat32;
    case XLLM_DTYPE_FLOAT64:
      return torch::kFloat64;
    case XLLM_DTYPE_BFLOAT16:
      return torch::kBFloat16;
    case XLLM_DTYPE_INT8:
      return torch::kInt8;
    case XLLM_DTYPE_INT16:
      return torch::kInt16;
    case XLLM_DTYPE_INT32:
      return torch::kInt32;
    case XLLM_DTYPE_INT64:
      return torch::kInt64;
    case XLLM_DTYPE_UINT8:
      return torch::kUInt8;
    case XLLM_DTYPE_UINT16:
      return torch::kUInt16;
    case XLLM_DTYPE_UINT32:
      return torch::kUInt32;
    case XLLM_DTYPE_UINT64:
      return torch::kUInt64;
    case XLLM_DTYPE_BOOL:
      return torch::kBool;
    case XLLM_DTYPE_STRING:
      throw std::runtime_error(
          "String dtype is not supported for torch::Tensor");
    default:
      throw std::runtime_error("Unsupported XLLM_DataType: " +
                               std::to_string(dtype));
  }
}

torch::Tensor convert_xllm_tensor_to_torch(const XLLM_Tensor& xllm_tensor) {
  if (xllm_tensor.data == nullptr) {
    throw std::runtime_error("XLLM_Tensor data pointer is null");
  }

  torch::ScalarType scalar_type =
      xllm_dtype_to_torch_scalar_type(xllm_tensor.dtype);

  std::vector<int64_t> shape;
  for (int i = 0; i < xllm_tensor.dims.rank; ++i) {
    int dim = xllm_tensor.dims.dim[i];
    if (dim > 0) {
      shape.push_back(dim);
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("XLLM_Tensor all dimensions are invalid value");
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(xllm_tensor.data), shape, scalar_type)
          .clone();

  return tensor;
}

xllm::MMDataItem convert_xllm_mm_item_to_internal(
    const XLLM_MM_Item& xllm_item) {
  uint32_t xllm_type_val = static_cast<uint32_t>(xllm_item.type);
  xllm::MMType::Value internal_val = xllm::MMType::NONE;

  switch (xllm_type_val) {
    case XLLM_MM_TYPE_EMBEDDING:
      internal_val = xllm::MMType::EMBEDDING;
      break;
    case XLLM_MM_TYPE_IMAGE:
      internal_val = xllm::MMType::IMAGE;
      break;
    case XLLM_MM_TYPE_VIDEO:
      internal_val = xllm::MMType::VIDEO;
      break;
    case XLLM_MM_TYPE_AUDIO:
      internal_val = xllm::MMType::AUDIO;
      break;
    case XLLM_MM_TYPE_NONE:
      internal_val = xllm::MMType::NONE;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported XLLM_MM_Type: ") +
                               std::to_string(xllm_type_val));
  }

  xllm::MMType item_type(internal_val);
  xllm::MMDataItem internal_item(item_type);

  xllm::MMItemState& state = internal_item.mutable_state();
  xllm::MMItemState::TokenPos& token_pos = state.mutable_token_pos();
  token_pos.offset = xllm_item.state.token_pos.offset;
  token_pos.length = xllm_item.state.token_pos.length;

  if (xllm_item.data.is_single_tensor) {
    torch::Tensor tensor =
        convert_xllm_tensor_to_torch(xllm_item.data.data.tensor);
    internal_item.add("tensor", tensor);
  } else {
    std::vector<torch::Tensor> tensor_list;
    const XLLM_Tensors& xllm_tensors = xllm_item.data.data.tensors;
    for (size_t i = 0; i < xllm_tensors.entries_size; ++i) {
      tensor_list.push_back(
          convert_xllm_tensor_to_torch(xllm_tensors.entries[i]));
    }
    internal_item.add("tensor_list", tensor_list);
  }

  return internal_item;
}

bool convert_xllm_mm_data_to_internal(const XLLM_MM_Data* mm_data,
                                      xllm::MMData& internal_mm_data) {
  if (mm_data == nullptr || mm_data->type_mask == XLLM_MM_TYPE_NONE) {
    return false;
  }

  xllm::MMType::Value internal_val =
      static_cast<xllm::MMType::Value>(mm_data->type_mask);
  xllm::MMType mm_type(internal_val);

  if (mm_data->is_dict) {
    const XLLM_MM_Dict& xllm_dict = mm_data->data.dict;
    xllm::MMDict internal_dict;

    for (size_t i = 0; i < xllm_dict.entries_size; ++i) {
      const XLLM_MM_DictEntry& xllm_entry = xllm_dict.entries[i];
      xllm::MMKey key(xllm_entry.key);

      const XLLM_MM_Value& xllm_value = xllm_entry.value;
      if (xllm_value.is_single_tensor) {
        torch::Tensor tensor =
            convert_xllm_tensor_to_torch(xllm_value.data.tensor);
        internal_dict.insert({key, tensor});
      } else {
        std::vector<torch::Tensor> tensor_list;
        const XLLM_Tensors& xllm_tensors = xllm_value.data.tensors;
        for (size_t j = 0; j < xllm_tensors.entries_size; ++j) {
          tensor_list.push_back(
              convert_xllm_tensor_to_torch(xllm_tensors.entries[j]));
        }
        internal_dict.insert({key, tensor_list});
      }
    }

    internal_mm_data.set<xllm::MMDict>(mm_type, internal_dict);
  } else {
    const XLLM_MM_Items& xllm_items = mm_data->data.items;
    xllm::MMItemVec internal_item_vec;

    for (size_t i = 0; i < xllm_items.entries_size; ++i) {
      const XLLM_MM_Item& xllm_item = xllm_items.entries[i];

      xllm::MMDataItem internal_item =
          convert_xllm_mm_item_to_internal(xllm_item);
      internal_item_vec.push_back(std::move(internal_item));
    }

    internal_mm_data.set<xllm::MMItemVec>(mm_type, internal_item_vec);
  }

  return true;
}

// 1. LLM Handler + const char* (text completions)
template XLLM_Response* handle_inference_request<XLLM_LLM_Handler, const char*>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 2. LLM Handler + std::vector<xllm::Message> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_LLM_Handler, std::vector<xllm::Message>>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 3. REC Handler + const char* (REC completions)
template XLLM_Response* handle_inference_request<XLLM_REC_Handler, const char*>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 4. REC Handler + std::vector<xllm::Message> (REC chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<xllm::Message>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 5. REC Handler + std::vector<int> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<int>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<int>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);
}  // namespace helper
}  // namespace xllm
