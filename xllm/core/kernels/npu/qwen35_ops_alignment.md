# Qwen3.5 NPU Op Alignment

This note records the Qwen3.5 normal-inference operator alignment work against
`vllm-ascend`, limited to the LLM path.

## Migrated Triton Kernels

The migrated Triton kernels that are required for the current Qwen3.5 serving
path are:

1. `_causal_conv1d_update_kernel_npu_tiled`
2. `fused_recurrent_gated_delta_rule_fwd_kernel`
3. `fused_sigmoid_gating_delta_rule_update_kernel`

They are provided from `third_party/torch_npu_ops` and consumed by xLLM
through the NPU kernel wrappers in this directory.

## xLLM Call Path

`xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` uses:

- prefill:
  - `kernel::causal_conv1d(...)`
  - `kernel::chunk_gated_delta_rule(...)`
- decode:
  - `kernel::causal_conv1d_update(...)`
  - `kernel::fused_sigmoid_gating_delta_rule_update(...)`

For normal serving correctness, the prefill `chunk_gated_delta_rule(...)`
wrapper currently keeps the recurrent-equivalent implementation as the active
path. This keeps normal inference aligned with `vllm-ascend` output quality
without opening an incomplete chunk migration by default.

## Why The Stable Prefill Fallback Stays

`vllm-ascend` Qwen3.5 prefill uses the full chunk stack, not just one Triton
kernel. The current xLLM migration only needs the three kernels above for the
validated normal-inference path. The chunk stack still needs a dedicated
follow-up migration before it can safely replace the default fallback.

## Remaining Work For Full Chunk Alignment

If xLLM later needs to match the full `vllm-ascend` chunk prefill path by
default, the remaining stack to port and integrate is:

1. `chunk_local_cumsum`
2. `chunk_scaled_dot_kkt_fwd`
3. `solve_tril`
4. `recompute_w_u_fwd`
5. `chunk_gated_delta_rule_fwd_h`
6. `chunk_fwd_o`

Reference implementation:

- `vllm_ascend/ops/triton/fla/chunk.py`
- `vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py`
- `vllm_ascend/ops/triton/fla/solve_tril.py`
- `vllm_ascend/ops/triton/fla/wy_fast.py`
- `vllm_ascend/ops/triton/fla/chunk_delta_h.py`
- `vllm_ascend/ops/triton/fla/chunk_o.py`

## Recommended Integration Plan

To switch xLLM from the current stable fallback to the full chunk path, the
recommended sequence is:

1. Add the missing chunk kernels to `third_party/torch_npu_ops`.
   Keep the Triton binaries and launcher wrappers colocated with the existing
   Qwen3.5 kernels under `triton_npu/torch_api/`.
2. Expose the new launchers through the same xLLM NPU wrapper layers used by
   the current migrated kernels:
   - `xllm/core/kernels/npu/npu_ops_api.h`
   - `xllm/core/kernels/param.h`
   - `xllm/core/kernels/ops_api.h`
   - `xllm/core/kernels/ops_api.cpp`
3. Implement a dedicated chunk-stack wrapper in
   `xllm/core/kernels/npu/gated_delta_net.cpp` that mirrors
   `vllm_ascend/ops/triton/fla/chunk.py`:
   - `chunk_local_cumsum`
   - `chunk_scaled_dot_kkt_fwd`
   - `solve_tril`
   - `recompute_w_u_fwd`
   - `chunk_gated_delta_rule_fwd_h`
   - `chunk_fwd_o`
4. Only after numeric validation against the recurrent reference and
   `vllm-ascend` service output should the default prefill path be switched from
   the recurrent-equivalent fallback to the full chunk path.
