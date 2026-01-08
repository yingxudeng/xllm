#include <torch/all.h>

#include "cuda_ops_api.h"

// Declare the global function (defined in scaled_mm_entry.cu)
// Note: We use extern "C++" to ensure C++ linkage, which is default.
void cutlass_scaled_mm(torch::Tensor& c,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

namespace xllm::kernel::cuda {

void cutlass_scaled_mm(torch::Tensor& c,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
  // Call the global function
  ::cutlass_scaled_mm(c, a, b, a_scales, b_scales, bias);
}

}  // namespace xllm::kernel::cuda