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

  uint32_t beam_size = in_sequence_group.size(1);
  
  uint32_t top_k = top_tokens.size(1);
  uint32_t total_rounds = in_sequence_group.size(2);
  
  CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";
  
  if (current_step == 0) {
  
    // [batch_size, beam_size]
    auto tokens = top_tokens.view({batch_size * 1, top_k}).slice(1, 0, beam_size);
    // [batch_size * beam_size]

    tokens = tokens.reshape({batch_size * beam_size, 1});
    // [batch_size, beam_size]

    auto init_probs = top_logprobs.view({batch_size * 1, top_k}).slice(1, 0, beam_size);
    // [batch_size * beam_size]

    init_probs = init_probs.reshape({batch_size * beam_size, 1});

    out_acc_logprob.copy_(init_probs);
    out_token_ids.copy_(tokens);

    auto indices = torch::arange(beam_size, torch::kInt32).to(device); // [beam_size]
    indices = indices.unsqueeze(0).expand({batch_size, -1}).reshape({-1, 1}); // [batch * beam, 1]
    out_token_index.copy_(indices);

    auto sequence_view = out_sequence_group.view({batch_size * beam_size, total_rounds});

    sequence_view.select(1, 0).copy_(tokens.squeeze(1));
  
  } else {
  
    auto combined_probs = acc_logprob + top_logprobs;

    combined_probs = combined_probs.view({batch_size, beam_size * top_k});

    auto topk_result = torch::topk(combined_probs, beam_size, -1);
    auto new_probs = std::get<0>(topk_result);    // [batch_size, beam_size]
    auto new_indices = std::get<1>(topk_result);  // [batch_size, beam_size]

    auto parent_beam = (new_indices / top_k).to(torch::kLong);      // [batch_size, beam_size]
    auto token_in_beam = (new_indices % top_k).to(torch::kLong);    // [batch_size, beam_size]

    // [batch_size, 1]
    auto batch_idx = torch::arange(batch_size, torch::kLong).to(device).unsqueeze(1);
    // [batch_size, beam_size, top_k]
    auto top_tokens_reshaped = top_tokens.view({batch_size, beam_size, top_k});

    auto new_tokens = top_tokens_reshaped.index({batch_idx, parent_beam, token_in_beam});

    out_acc_logprob.copy_(new_probs.reshape({-1, 1}));
    out_token_index.copy_(new_indices.reshape({-1, 1}));
    out_token_ids.copy_(new_tokens.reshape({-1, 1}));

    auto batch_range = torch::arange(batch_size, torch::kInt32).to(device).unsqueeze(1).expand({-1, beam_size});
    auto beam_range = torch::arange(beam_size, torch::kInt32).to(device).unsqueeze(0).expand({batch_size, -1});

    out_sequence_group.slice(2, 0, current_step) = 
      in_sequence_group.index({batch_range, parent_beam, torch::indexing::Slice(0, current_step)});

    out_sequence_group.slice(2, current_step, current_step + 1) = new_tokens.unsqueeze(2);
  }
}

TEST_F(BeamSearchTest, CorrectnessTest) {
  // Small shapes are enough to catch indexing bugs, while keeping the test
  // fast.
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t top_k = 2;
  const int64_t total_rounds = 3;
  // int32_t current_step = 0;
  auto fp_options =
        torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto int_options =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);
  (void)dtype_;  // test uses explicit dtypes per tensor

  torch::Tensor acc_logprob;
  torch::Tensor in_sequence_group;
  torch::Tensor top_tokens;
  torch::Tensor top_logprobs;
  torch::Tensor out_acc_logprob;
  torch::Tensor out_token_ids;
  torch::Tensor out_token_index;
  torch::Tensor out_beam_count_prefix_sums;
  torch::Tensor out_sequence_group;

  acc_logprob = torch::randn({batch_size * beam_size, 1}, fp_options);
  in_sequence_group = torch::randint(0, 1000, {batch_size, beam_size, total_rounds}, int_options);

  out_acc_logprob = torch::zeros({batch_size * beam_size, 1}, fp_options);
  out_token_ids = torch::zeros({batch_size * beam_size, 1}, int_options);
  out_token_index = torch::zeros({batch_size * beam_size, 1}, int_options);
  out_beam_count_prefix_sums = torch::zeros({batch_size * beam_size, 1}, int_options);
  out_sequence_group = torch::zeros({batch_size, beam_size, total_rounds}, int_options);

  
  for (int32_t current_step = 0; current_step < total_rounds; current_step++) {
    LOG(INFO) << "--------------------------------step " << current_step
              << "--------------------------------";

    if (current_step == 0) {
      top_tokens =
          torch::randint(0, 1000, {batch_size, 1, top_k}, int_options);
      top_logprobs =
          torch::randn({batch_size, 1, top_k}, fp_options);

    } else {
      top_tokens =
          torch::randint(0, 1000, {batch_size, beam_size, top_k}, int_options);
      top_logprobs =
          torch::randn({batch_size, beam_size, top_k}, fp_options);

    }

    top_tokens = top_tokens.view({-1, top_k});
    top_logprobs = top_logprobs.view({-1, top_k});
    LOG(INFO) << "top_tokens: " << top_tokens;
    LOG(INFO) << "top_logprobs: " << top_logprobs;
    
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

    acc_logprob = out_acc_logprob;
    in_sequence_group = out_sequence_group;
  }
}