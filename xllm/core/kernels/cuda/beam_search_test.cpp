#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>
class BeamSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
    dtype_ = torch::kFloat16;
  }

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat16;
};

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 int32_t batch_size,
                 int32_t current_step) {
  torch::Device device = acc_logprob.device();

  int32_t beam_size = in_sequence_group.size(1);

  int64_t top_k = top_tokens.size(2);
  uint32_t total_rounds = in_sequence_group.size(2);

  CHECK_GE(top_k, 1) << "top_k must be >= 1.";

  if (current_step == 0) {
    auto tokens_flat = top_tokens.view({batch_size * beam_size, -1});
    // [batch * beam]
    auto first_col = tokens_flat.select(1, 0).to(torch::kInt32);

    auto sequence_view =
        out_sequence_group.view({batch_size * beam_size, total_rounds});

    // Copy new token to position 0
    sequence_view.select(1, 0).copy_(first_col.view({batch_size * beam_size}));

    out_token_ids.copy_(first_col.view({batch_size, beam_size}));

    auto indices =
        torch::arange(batch_size * beam_size, torch::kInt32).to(device);
    out_token_index.copy_(indices.view({batch_size * beam_size, 1}));

    // For step 0 in this logic, log probs are initialized to -inf if not
    // provided, but here we might want to copy from top_logprobs if available
    // or just zero. Following Python logic exactly:
    out_acc_logprob.fill_(-std::numeric_limits<float>::infinity());
    out_beam_count_prefix_sums.zero_();

  } else {
    // Logic adapted from BeamSearchTorch::process

    // 1. Calculate candidate scores
    // acc_logprob: [batch, beam] -> [batch, beam, 1] for broadcasting
    // top_logprobs: [batch, beam, top_k]
    auto candidate_scores =
        acc_logprob.unsqueeze(2) + top_logprobs;  // [batch, beam, top_k]

    // 2. Reshape to [batch, beam * top_k]
    candidate_scores = candidate_scores.view({batch_size, beam_size * top_k});

    // 3. TopK selection
    auto topk_result = torch::topk(candidate_scores, beam_size, 1, true, true);
    auto topk_scores = std::get<0>(topk_result);   // [batch, beam]
    auto topk_indices = std::get<1>(topk_result);  // [batch, beam]

    // 4. Decode indices
    auto selected_beam =
        torch::div(topk_indices, top_k, "floor");  // [batch, beam]
    auto selected_within_top =
        torch::remainder(topk_indices, top_k);  // [batch, beam]

    // 5. Calculate global parent indices
    auto batch_idx = torch::arange(batch_size, torch::kLong)
                         .to(device)
                         .unsqueeze(1);  // [batch, 1]
    auto global_parent_indices =
        batch_idx * beam_size + selected_beam;  // [batch, beam]

    // 6. Select tokens using global indices
    // Flatten top_tokens to [batch * beam, top_k] to use index_select with
    // global indices
    auto top_tokens_flat = top_tokens.view({-1, top_k});  // [batch*beam, top_k]
    // Select the row corresponding to the parent beam
    auto selected_top = top_tokens_flat.index_select(
        0, global_parent_indices.view(-1));  // [batch*beam, top_k]

    // Gather the specific token from top_k candidates
    auto selected_within_top_flat =
        selected_within_top.view({-1, 1}).to(torch::kLong);
    auto next_tokens =
        selected_top.gather(1, selected_within_top_flat);  // [batch*beam, 1]

    // 7. Update outputs
    out_acc_logprob.copy_(topk_scores);
    out_token_ids.copy_(next_tokens.view({batch_size, beam_size}));
    out_token_index.copy_(
        global_parent_indices.view({-1, 1}).to(torch::kInt32));

    // 8. Update Sequence Group (History)
    // Copy history from parents
    auto in_seq_flat = in_sequence_group.view({-1, total_rounds});
    auto out_seq_flat = out_sequence_group.view({-1, total_rounds});
    auto selected_history =
        in_seq_flat.index_select(0, global_parent_indices.view(-1));

    using namespace torch::indexing;
    out_seq_flat.index_put_(
        {Slice(), Slice(0, current_step)},
        selected_history.index({Slice(), Slice(0, current_step)}));
    out_seq_flat.index_put_({Slice(), current_step}, next_tokens.view(-1));
  }
}

TEST_F(BeamSearchTest, CorrectnessTest) {
  // Small shapes are enough to catch indexing bugs, while keeping the test
  // fast.
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t top_k = 2;
  const int64_t total_rounds = 3;
  int32_t current_step = 0;
  (void)dtype_;  // test uses explicit dtypes per tensor
  for (int32_t current_step = 0; current_step < total_rounds; current_step++) {
    auto fp_options =
        torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    auto int_options =
        torch::TensorOptions().device(device_).dtype(torch::kInt32);

    torch::Tensor acc_logprob =
        torch::randn({batch_size, beam_size}, fp_options);
    torch::Tensor in_sequence_group = torch::randint(
        0, 1000, {batch_size, beam_size, total_rounds}, int_options);
    torch::Tensor top_tokens;
    if (current_step == 0) {
      top_tokens =
          torch::randint(0, 1000, {batch_size, beam_size, 1}, int_options);
    } else {
      top_tokens =
          torch::randint(0, 1000, {batch_size, beam_size, top_k}, int_options);
    }
    torch::Tensor top_logprobs =
        torch::randn({batch_size, beam_size, top_k}, fp_options);

    torch::Tensor out_acc_logprob =
        torch::zeros({batch_size, beam_size}, fp_options);
    torch::Tensor out_token_ids =
        torch::zeros({batch_size, beam_size}, int_options);
    // NOTE: shape must be [B*BEAM, 1] to match copy_ sites inside
    // beam_search().
    torch::Tensor out_token_index =
        torch::zeros({batch_size * beam_size, 1}, int_options);
    torch::Tensor out_beam_count_prefix_sums =
        torch::zeros({batch_size * beam_size, 1}, int_options);
    torch::Tensor out_sequence_group =
        torch::zeros({batch_size, beam_size, total_rounds}, int_options);
    LOG(INFO) << "--------------------------------step " << current_step
              << "--------------------------------";
    beam_search(acc_logprob,
                in_sequence_group,
                top_tokens,
                top_logprobs,
                out_acc_logprob,
                out_token_ids,
                out_token_index,
                out_beam_count_prefix_sums,
                out_sequence_group,
                batch_size,
                current_step);
    LOG(INFO) << "out_acc_logprob: " << out_acc_logprob;
    LOG(INFO) << "out_token_ids: " << out_token_ids;
    LOG(INFO) << "out_token_index: " << out_token_index;
    LOG(INFO) << "out_beam_count_prefix_sums: " << out_beam_count_prefix_sums;
    LOG(INFO) << "out_sequence_group: " << out_sequence_group;
  }
}