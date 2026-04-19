#!/usr/bin/env python3
"""Measure dispatch overhead breakdown: Python→Cython→C→ACL."""
import os
import sys
import time

os.chdir("/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/xllm")
sys.path.insert(0, "/export/home/dengyingxu1/projects/xllm/third_party/tilelang-ascend")

import torch
import torch_npu

from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
    fused_gdn_gating_kernel_jit,
)

batch_size = 4096
num_heads = 32
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

# Warmup
for _ in range(10):
    kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
torch.npu.synchronize()

# Test 1: E2E timing with sync (includes dispatch + device)
iters = 200
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
torch.npu.synchronize()
e2e_sync = (time.perf_counter() - t0) / iters * 1e6

# Test 2: E2E timing WITHOUT sync (just host-side cost)
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    kernel(A_log, a, b, dt_bias, g_out, beta_out, batch_size, 1.0, 20.0)
host_only = (time.perf_counter() - t0) / iters * 1e6
torch.npu.synchronize()

# Test 3: naive PyTorch for comparison
def naive_forward():
    x = a.to(torch.float32) + dt_bias
    g = -A_log.exp() * torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)
    beta = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
    return g, beta

for _ in range(10):
    naive_forward()
torch.npu.synchronize()

torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    naive_forward()
torch.npu.synchronize()
naive_sync = (time.perf_counter() - t0) / iters * 1e6

torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    naive_forward()
naive_host = (time.perf_counter() - t0) / iters * 1e6
torch.npu.synchronize()

# Test 4: raw ctypes call overhead (bypass Cython wrapper)
import ctypes
adapter = kernel.adapter
lib = adapter.lib
stream = torch.npu.current_stream().npu_stream

call_args_cached = [
    ctypes.c_void_p(A_log.data_ptr()),
    ctypes.c_void_p(a.data_ptr()),
    ctypes.c_void_p(b.data_ptr()),
    ctypes.c_void_p(dt_bias.data_ptr()),
    ctypes.c_void_p(g_out.data_ptr()),
    ctypes.c_void_p(beta_out.data_ptr()),
    batch_size,
    ctypes.c_float(1.0),
    ctypes.c_float(20.0),
    ctypes.c_void_p(stream),
]

# warmup direct call
for _ in range(10):
    lib.call(*call_args_cached)
torch.npu.synchronize()

# direct call with sync
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    lib.call(*call_args_cached)
torch.npu.synchronize()
direct_sync = (time.perf_counter() - t0) / iters * 1e6

# direct call without sync
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    lib.call(*call_args_cached)
direct_host = (time.perf_counter() - t0) / iters * 1e6
torch.npu.synchronize()

print(f"=== B={batch_size}, H={num_heads} dispatch overhead breakdown ===")
print(f"")
print(f"Fused kernel (via Cython wrapper):")
print(f"  E2E with sync:    {e2e_sync:.1f}us")
print(f"  Host-only (no sync): {host_only:.1f}us")
print(f"  Device time (approx): {e2e_sync - host_only:.1f}us")
print(f"")
print(f"Fused kernel (direct ctypes lib.call, bypass Cython):")
print(f"  E2E with sync:    {direct_sync:.1f}us")
print(f"  Host-only (no sync): {direct_host:.1f}us")
print(f"  Device time (approx): {direct_sync - direct_host:.1f}us")
print(f"  Cython wrapper overhead: {e2e_sync - direct_sync:.1f}us")
print(f"")
print(f"Naive PyTorch (7 CANN ops):")
print(f"  E2E with sync:    {naive_sync:.1f}us")
print(f"  Host-only (no sync): {naive_host:.1f}us")
print(f"")
print(f"Key insight:")
print(f"  Fused dispatch overhead: {host_only:.1f}us (host) vs Naive: {naive_host:.1f}us (host)")
print(f"  If fused host cost were same as naive: {naive_host + (e2e_sync - host_only):.1f}us total")
