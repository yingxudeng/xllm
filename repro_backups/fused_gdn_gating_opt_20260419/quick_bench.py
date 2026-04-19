#!/usr/bin/env python3
"""Quick bench: fused vs naive for H=32, specific batch sizes."""

import sys
import time

import os
os.chdir("/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/third_party/tilelang-ascend")

import torch
import torch_npu

from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
    fused_gdn_gating_kernel_jit,
)


def bench_fused(jit_fn, batch_size, num_heads=32, warmup=10, iters=100):
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    kernel = jit_fn(num_batches=batch_size, compile_max_batch=262144, num_heads=num_heads)
    for _ in range(warmup):
        kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def bench_naive(batch_size, num_heads=32, warmup=10, iters=100):
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)

    def run():
        x = a.to(torch.float32) + dt_bias
        g = -A_log.exp() * torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)
        beta = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
        return g, beta

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


jit_fn = fused_gdn_gating_kernel_jit
batches = [16, 48, 256, 1024, 4096, 16384, 65536, 262144]

print(f"{'batch':>10} {'fused_us':>10} {'naive_us':>10} {'speedup':>8} {'fused_dev':>10} {'dispatch':>10}")
print("-" * 62)
for bs in batches:
    fused_us = bench_fused(jit_fn, bs)
    naive_us = bench_naive(bs)
    speedup = naive_us / fused_us
    print(f"{bs:>10} {fused_us:>10.1f} {naive_us:>10.1f} {speedup:>7.2f}x")
