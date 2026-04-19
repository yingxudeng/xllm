#!/usr/bin/env python3
"""Dump the generated C++ source code for the fused_gdn_gating kernel."""
import os
import sys

os.chdir("/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/third_party/tilelang-ascend")

from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
    fused_gdn_gating_kernel_jit,
)

kernel = fused_gdn_gating_kernel_jit(
    num_batches=4096,
    compile_max_batch=262144,
    num_heads=32,
)

# Print the wrapped source (host call function)
print("=== WRAPPED SOURCE (host call) ===")
print(kernel.get_kernel_source())

print("\n=== SRCPATH ===")
print(kernel.srcpath)
