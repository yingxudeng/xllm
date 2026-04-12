/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <vector>

#include "xllm/core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

std::tuple<torch::Tensor, torch::Tensor> causal_conv1d_expected(
    torch::Tensor x,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& initial_state,
    bool activation) {
  torch::ScalarType input_dtype = x.scalar_type();
  int64_t width = weight.size(1);
  torch::Tensor x_conv = x.to(weight.dtype());
  torch::Tensor out;
  std::vector<int64_t> stride = {1};
  std::vector<int64_t> no_padding = {0};
  std::vector<int64_t> dilation = {1};
  std::vector<int64_t> padding = {width - 1};
  if (initial_state.has_value()) {
    torch::Tensor padded_input =
        torch::cat({initial_state.value().to(x_conv.dtype()), x_conv}, -1);
    out = torch::conv1d(padded_input,
                        weight.unsqueeze(1),
                        bias,
                        stride,
                        no_padding,
                        dilation,
                        x_conv.size(1));
  } else {
    out = torch::conv1d(x_conv,
                        weight.unsqueeze(1),
                        bias,
                        stride,
                        padding,
                        dilation,
                        x_conv.size(1));
  }
  out = out.slice(-1, 0, x.size(-1));
  if (activation) {
    out = torch::silu(out);
  }

  torch::Tensor state_source = initial_state.has_value()
                                   ? torch::cat({initial_state.value(), x}, -1)
                                   : x;
  torch::Tensor final_state =
      torch::nn::functional::pad(
          state_source.slice(
              -1,
              std::max<int64_t>(state_source.size(-1) - (width - 1),
                                static_cast<int64_t>(0)),
              state_source.size(-1)),
          torch::nn::functional::PadFuncOptions(
              {std::max<int64_t>((width - 1) - state_source.size(-1), 0), 0}))
          .to(input_dtype);
  return std::make_tuple(out.to(input_dtype), final_state);
}

TEST(CausalConv1dTest, MatchesDepthwiseReferenceAndUpdatesCache) {
  torch::manual_seed(0);

  int64_t batch_size = 3;
  int64_t channels = 5;
  int64_t width = 4;
  int64_t max_seq_len = 6;
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);

  torch::Tensor x = torch::randn({batch_size, channels, max_seq_len}, opts);
  torch::Tensor weight = torch::randn({channels, width}, opts);
  torch::Tensor bias = torch::randn({channels}, opts);
  torch::Tensor seq_lens = torch::tensor(std::vector<int64_t>{6, 4, 2});
  torch::Tensor conv_state_source =
      torch::randn({7, channels, width - 1}, opts);
  torch::Tensor conv_state_before = conv_state_source.clone();
  torch::Tensor conv_state_indices =
      torch::tensor(std::vector<int64_t>{3, 5, 1}, torch::kInt64);
  torch::Tensor has_initial_state =
      torch::tensor(std::vector<int64_t>{1, 0, 1},
                    torch::TensorOptions().dtype(torch::kBool));

  torch::Tensor actual = causal_conv1d(x,
                                       weight,
                                       bias,
                                       seq_lens,
                                       conv_state_source,
                                       conv_state_indices,
                                       has_initial_state,
                                       true,
                                       -1);

  torch::Tensor expected = torch::zeros_like(actual);
  torch::Tensor expected_state = conv_state_before.clone();
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int64_t valid_len = seq_lens[batch_idx].item<int64_t>();
    int64_t cache_idx = conv_state_indices[batch_idx].item<int64_t>();
    std::optional<torch::Tensor> initial_state = std::nullopt;
    if (has_initial_state[batch_idx].item<bool>()) {
      initial_state =
          conv_state_before.index({cache_idx}).unsqueeze(0).contiguous();
    }
    auto x_slice =
        x.index({batch_idx}).slice(-1, 0, valid_len).unsqueeze(0).contiguous();
    auto [expected_slice, expected_final_state] =
        causal_conv1d_expected(x_slice, weight, bias, initial_state, true);
    expected.index_put_({batch_idx,
                         torch::indexing::Slice(),
                         torch::indexing::Slice(0, valid_len)},
                        expected_slice.squeeze(0));
    expected_state.index_put_({cache_idx}, expected_final_state.squeeze(0));
  }

  EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(conv_state_source, expected_state, 1e-5, 1e-5));
}

}  // namespace
}  // namespace xllm::kernel::npu
