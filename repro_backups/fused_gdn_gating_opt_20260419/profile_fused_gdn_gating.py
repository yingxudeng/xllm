#!/usr/bin/env python3
"""Standalone fused_gdn_gating kernel profiling helper.

Run inside dyx-xllm-cann85-main with PROFILING_MODE=dynamic.
Supports --wait-for-attach to pause before profiled iterations so msprof
can attach and start collecting.

Usage:
  # Direct run (no msprof):
  python profile_fused_gdn_gating.py --batch-size 262144 --num-heads 32

  # With msprof attach (see run_profile_fused_gdn_gating.sh):
  python profile_fused_gdn_gating.py --batch-size 262144 --num-heads 32 \
    --wait-for-attach --signal-file /tmp/msprof_ready
"""

import argparse
import os
import sys
import time
from pathlib import Path


XLLM_ROOT = "/export/home/dengyingxu1/projects/xllm"


def get_kernel_module():
    if XLLM_ROOT not in sys.path:
        sys.path.insert(0, XLLM_ROOT)
    from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
        fused_gdn_gating_kernel_jit,
    )

    return fused_gdn_gating_kernel_jit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--softplus-beta", type=float, default=1.0)
    parser.add_argument("--softplus-threshold", type=float, default=20.0)
    parser.add_argument(
        "--wait-for-attach",
        action="store_true",
        help="After warmup, pause and wait for signal file before profiled iterations.",
    )
    parser.add_argument(
        "--signal-file",
        type=str,
        default="/tmp/fused_gdn_gating_msprof_ready",
        help="File whose creation signals that msprof is attached and profiled iters can begin.",
    )
    args = parser.parse_args()

    import torch

    device = torch.device("npu")
    jit_fn = get_kernel_module()

    batch_size = args.batch_size
    num_heads = args.num_heads
    compile_max_batch = 262144

    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((batch_size, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty((batch_size, num_heads), device=device, dtype=torch.bfloat16)

    print(f"[profile] JIT compiling kernel bs={batch_size} nh={num_heads} ...")
    kernel = jit_fn(
        num_batches=batch_size,
        compile_max_batch=compile_max_batch,
        num_heads=num_heads,
    )
    print("[profile] JIT done.")

    print(f"[profile] Warmup {args.warmup} iterations ...")
    for _ in range(args.warmup):
        kernel(
            A_log, a, b, dt_bias, g_out, beta_out,
            batch_size, args.softplus_beta, args.softplus_threshold,
        )
    torch.npu.synchronize()
    print("[profile] Warmup done.")

    pid = os.getpid()
    print(f"[profile] PID={pid}")

    if args.wait_for_attach:
        signal_path = Path(args.signal_file)
        if signal_path.exists():
            signal_path.unlink()
        print(f"[profile] Waiting for signal file: {args.signal_file}")
        print(f"[profile] Attach msprof to PID={pid}, then create the signal file.")
        while not signal_path.exists():
            time.sleep(0.5)
        print("[profile] Signal received, starting profiled iterations.")

    print(f"[profile] Running {args.iters} profiled iterations ...")
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        kernel(
            A_log, a, b, dt_bias, g_out, beta_out,
            batch_size, args.softplus_beta, args.softplus_threshold,
        )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    avg_us = elapsed / args.iters * 1e6

    print(f"[profile] {args.iters} iters in {elapsed:.4f}s, avg {avg_us:.2f} us/iter")
    print("PROFILING DONE")


if __name__ == "__main__":
    main()
