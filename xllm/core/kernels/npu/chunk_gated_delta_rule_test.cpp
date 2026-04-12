#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <optional>

#include "xllm/core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

torch::Tensor L2Norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  torch::Tensor norm =
      torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

std::tuple<torch::Tensor, torch::Tensor> RecurrentReference(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    const std::optional<torch::Tensor>& initial_state,
    bool use_qk_l2norm_in_kernel) {
  torch::ScalarType initial_dtype = query.scalar_type();

  if (use_qk_l2norm_in_kernel) {
    query = L2Norm(query, -1, 1e-6);
    key = L2Norm(key, -1, 1e-6);
  }

  auto to_float32_and_transpose = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_float32_and_transpose(query);
  key = to_float32_and_transpose(key);
  value = to_float32_and_transpose(value);
  beta = to_float32_and_transpose(beta);
  g = to_float32_and_transpose(g);

  int64_t batch_size = key.size(0);
  int64_t num_k_heads = key.size(1);
  int64_t num_v_heads = value.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);
  int64_t kv_head_ratio = num_v_heads / num_k_heads;

  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;

  torch::Tensor output = torch::zeros(
      {batch_size, num_v_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_state;
  if (initial_state.has_value()) {
    last_state = initial_state.value().to(value.device(), torch::kFloat32);
  } else {
    last_state = torch::zeros(
        {batch_size, num_v_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  for (int64_t idx = 0; idx < sequence_length; ++idx) {
    torch::Tensor q_t = query.select(2, idx);
    torch::Tensor k_t = key.select(2, idx);
    if (kv_head_ratio > 1) {
      q_t = q_t.repeat_interleave(kv_head_ratio, 1);
      k_t = k_t.repeat_interleave(kv_head_ratio, 1);
    }
    torch::Tensor v_t = value.select(2, idx);
    torch::Tensor g_t = g.select(2, idx).exp().unsqueeze(-1).unsqueeze(-1);
    torch::Tensor beta_t = beta.select(2, idx).unsqueeze(-1);
    last_state = last_state * g_t;
    torch::Tensor kv_mem = torch::sum(last_state * k_t.unsqueeze(-1), -2);
    torch::Tensor delta = (v_t - kv_mem) * beta_t;
    last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    output.select(2, idx) = torch::sum(last_state * q_t.unsqueeze(-1), -2);
  }

  output = output.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(output, last_state);
}

TEST(ChunkGatedDeltaRuleTest, StableChunkWrapperMatchesRecurrentReference) {
  torch::manual_seed(0);

  constexpr int64_t batch_size = 2;
  constexpr int64_t max_seq_len = 128;
  constexpr int64_t num_qk_heads = 4;
  constexpr int64_t num_v_heads = 8;
  constexpr int64_t k_head_dim = 64;
  constexpr int64_t v_head_dim = 64;

  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor q = torch::randn(
      {batch_size, max_seq_len, num_qk_heads, k_head_dim}, float_opts);
  torch::Tensor k = torch::randn(
      {batch_size, max_seq_len, num_qk_heads, k_head_dim}, float_opts);
  torch::Tensor v = torch::randn(
      {batch_size, max_seq_len, num_v_heads, v_head_dim}, float_opts);
  torch::Tensor g = torch::nn::functional::logsigmoid(
      torch::randn({batch_size, max_seq_len, num_v_heads}, float_opts));
  torch::Tensor beta = torch::sigmoid(
      torch::randn({batch_size, max_seq_len, num_v_heads}, float_opts));
  torch::Tensor seq_lens =
      torch::tensor(std::vector<int64_t>{max_seq_len, 97}, torch::kInt64);
  torch::Tensor initial_state = torch::randn(
      {batch_size, num_v_heads, k_head_dim, v_head_dim}, float_opts);

  auto [chunk_output, chunk_final_state] = chunk_gated_delta_rule(
      q, k, v, g, beta, seq_lens, 64, initial_state, true, true);

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int64_t valid_len = seq_lens[batch_idx].item<int64_t>();
    auto q_slice =
        q.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto k_slice =
        k.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto v_slice =
        v.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto g_slice =
        g.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto beta_slice = beta.index({batch_idx})
                          .slice(0, 0, valid_len)
                          .unsqueeze(0)
                          .contiguous();
    auto initial_state_slice =
        initial_state.index({batch_idx}).unsqueeze(0).contiguous();

    auto [reference_output, reference_final_state] =
        RecurrentReference(q_slice,
                           k_slice,
                           v_slice,
                           g_slice,
                           beta_slice,
                           initial_state_slice,
                           true);

    EXPECT_TRUE(
        torch::allclose(chunk_output.index({batch_idx}).slice(0, 0, valid_len),
                        reference_output.squeeze(0),
                        1e-4,
                        1e-4));
    EXPECT_TRUE(torch::allclose(chunk_final_state.index({batch_idx}),
                                reference_final_state.squeeze(0),
                                1e-4,
                                1e-4));
    EXPECT_TRUE(torch::allclose(
        chunk_output.index({batch_idx}).slice(0, valid_len, max_seq_len),
        torch::zeros_like(
            chunk_output.index({batch_idx}).slice(0, valid_len, max_seq_len)),
        0.0,
        0.0));
  }
}

}  // namespace
}  // namespace xllm::kernel::npu
