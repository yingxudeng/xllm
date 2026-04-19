#!/usr/bin/env python3
"""Benchmark fused_gdn_gating TileLang kernel across batch sizes and head counts."""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


XLLM_ROOT = "/export/home/dengyingxu1/projects/xllm"


def get_kernel_module():
    if XLLM_ROOT not in sys.path:
        sys.path.insert(0, XLLM_ROOT)
    from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
        fused_gdn_gating_kernel_jit,
    )

    return fused_gdn_gating_kernel_jit


def bench_one(
    jit_fn,
    batch_size: int,
    num_heads: int,
    warmup: int,
    iters: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.bfloat16)

    kernel = jit_fn(
        num_batches=batch_size,
        compile_max_batch=262144,
        num_heads=num_heads,
    )

    for _ in range(warmup):
        kernel(
            A_log, a, b, dt_bias, g_out, beta_out,
            batch_size, softplus_beta, softplus_threshold,
        )
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel(
            A_log, a, b, dt_bias, g_out, beta_out,
            batch_size, softplus_beta, softplus_threshold,
        )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6
    throughput = batch_size / (elapsed / iters)
    return avg_us, throughput


def bench_naive_pytorch(
    batch_size: int,
    num_heads: int,
    warmup: int,
    iters: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    device = torch.device("npu")
    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)

    def naive_forward():
        x = a.to(torch.float32) + dt_bias
        g = -A_log.exp() * torch.nn.functional.softplus(
            x, beta=softplus_beta, threshold=softplus_threshold
        )
        beta = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
        return g, beta

    for _ in range(warmup):
        naive_forward()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        naive_forward()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024,
                 4096, 8192, 32768, 65536, 131072, 262144],
    )
    parser.add_argument(
        "--num-heads-list",
        type=int,
        nargs="+",
        default=[32, 128],
    )
    args = parser.parse_args()

    jit_fn = get_kernel_module()

    results = []
    header = f"{'batch':>10} {'heads':>6} {'fused_us':>12} {'naive_us':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for num_heads in args.num_heads_list:
        for batch_size in args.batch_sizes:
            try:
                fused_us, _ = bench_one(
                    jit_fn,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                naive_us = bench_naive_pytorch(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                speedup = naive_us / fused_us if fused_us > 0 else 0
                row = {
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "fused_us": f"{fused_us:.2f}",
                    "naive_us": f"{naive_us:.2f}",
                    "speedup": f"{speedup:.2f}x",
                }
                results.append(row)
                print(
                    f"{batch_size:>10} {num_heads:>6} {fused_us:>12.2f} {naive_us:>12.2f} {speedup:>7.2f}x"
                )
            except Exception as e:
                print(
                    f"{batch_size:>10} {num_heads:>6} {'ERROR':>12} {str(e)[:40]}"
                )
                results.append({
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "fused_us": "ERROR",
                    "naive_us": "ERROR",
                    "speedup": str(e)[:40],
                })

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["batch_size", "num_heads", "fused_us", "naive_us", "speedup"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
