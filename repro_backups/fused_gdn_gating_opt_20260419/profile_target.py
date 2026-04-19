#!/usr/bin/env python3
"""Profile naive PyTorch baseline for msprof comparison."""
import os
import sys
import time
import argparse

os.chdir("/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/third_party/tilelang-ascend")

import torch
import torch_npu


def run_naive(batch_size, num_heads, warmup, iters):
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)

    def naive_forward():
        x = a.to(torch.float32) + dt_bias
        g = -A_log.exp() * torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)
        beta = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
        return g, beta

    for _ in range(warmup):
        naive_forward()
    torch.npu.synchronize()

    for _ in range(iters):
        naive_forward()
    torch.npu.synchronize()


def run_fused(batch_size, num_heads, warmup, iters):
    from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
        fused_gdn_gating_kernel_jit,
    )
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.bfloat16)

    kernel = fused_gdn_gating_kernel_jit(
        num_batches=batch_size, compile_max_batch=262144, num_heads=num_heads,
    )

    for _ in range(warmup):
        kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
    torch.npu.synchronize()

    for _ in range(iters):
        kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
    torch.npu.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "fused"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(f"Running {args.mode} mode: B={args.batch_size}, H={args.num_heads}, "
          f"warmup={args.warmup}, iters={args.iters}")

    if args.mode == "naive":
        run_naive(args.batch_size, args.num_heads, args.warmup, args.iters)
    else:
        run_fused(args.batch_size, args.num_heads, args.warmup, args.iters)

    print("Done.")
