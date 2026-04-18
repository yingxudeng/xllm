#!/usr/bin/env python3

"""
Run the TileLang and Triton fused_gdn_gating kernels on the Python path.

This file is intentionally limited to kernel execution:
  - normal mode: compare Python-side launch timing with torch.npu.Event
  - worker mode: run warmup + measure loops without any profiling logic

The msprof orchestration and task-duration extraction live in a separate file:
  kernel_msprof_task_duration.py
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[6]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import triton
import triton.language as tl

from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import (
    DEFAULT_MAX_BATCH,
    DEFAULT_MAX_HEADS,
    SUPPORTED_NUM_HEADS,
    detect_vec_core_num,
    fused_gdn_gating_kernel_jit,
    select_launch_block_num,
)

UNIFIED_BUFFER_SIZE = 1572864
DEFAULT_NUM_BATCHES_LIST = (1, 2, 4, 8, 16, 32, 48, 1024, 4096, 16384, 65536, 262144)
DEFAULT_NUM_HEADS_LIST = (32, 128)
DEFAULT_WARMUP_ITERS = 20
DEFAULT_MEASURE_ITERS = 200
DEFAULT_SOFTPLUS_BETA = 1.0
DEFAULT_SOFTPLUS_THRESHOLD = 20.0
DEFAULT_SEED = 42


def get_vectorcore_num() -> int:
    props = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    return int(props["num_vectorcore"])


@triton.jit
def vllm_fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    COL_ITER: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER: tl.constexpr,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    for row_idx in range(0, ROW_ITER):
        batch_off = (
            i_b * ROW_ITER * BLK_BATCHES
            + row_idx * BLK_BATCHES
            + tl.arange(0, BLK_BATCHES)
        )

        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)
            off = (
                batch_off[:, None] * seq_len * NUM_HEADS
                + i_s * NUM_HEADS
                + head_off[None, :]
            )
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)

            blk_A_log = tl.load(A_log + head_off, mask=head_mask)
            blk_a = tl.load(a + off, mask=mask)
            blk_b = tl.load(b + off, mask=mask)
            blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(
                beta * x <= threshold,
                (1 / beta) * tl.log(1 + tl.exp(beta * x)),
                x,
            )

            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
            tl.store(
                beta_output + off,
                blk_beta_output.to(beta_output.dtype.element_ty),
                mask=mask,
            )


def run_vllm_triton(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    g: torch.Tensor,
    beta_output: torch.Tensor,
    beta: float,
    threshold: float,
) -> None:
    batch, num_heads = a.shape
    seq_len = 1

    num_cores = get_vectorcore_num()
    blk_heads = 8
    col_iter = triton.cdiv(num_heads, blk_heads)

    elem_size = a.element_size()
    max_ub_batches = int((UNIFIED_BUFFER_SIZE * 0.95) / (blk_heads * elem_size))
    if batch <= num_cores:
        progs = batch
        blk_batches = 1
        row_iter = 1
    else:
        progs = num_cores
        factor = 8 * num_heads
        calc_blk_batches = (
            triton.next_power_of_2(
                triton.cdiv(
                    int(UNIFIED_BUFFER_SIZE * 0.95),
                    factor * blk_heads * elem_size,
                )
            )
            // 2
        )
        blk_batches = max(1, min(calc_blk_batches, max_ub_batches, 64))
        row_per_core = triton.cdiv(batch, progs)
        row_iter = triton.cdiv(row_per_core, blk_batches)

    grid = (progs, seq_len)
    vllm_fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        batch,
        beta,
        threshold,
        BLK_HEADS=blk_heads,
        COL_ITER=col_iter,
        BLK_BATCHES=blk_batches,
        ROW_ITER=row_iter,
    )


def measure_npu_event_ms(fn, warmup_iters: int, measure_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(measure_iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return float(start.elapsed_time(end) / measure_iters)


def make_case_tensors(
    *,
    num_batches: int,
    num_heads: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("npu")

    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)

    tl_g = torch.empty((num_batches, num_heads), device=device, dtype=torch.float32)
    tl_beta = torch.empty((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    return {
        "A_log": A_log,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "tl_g": tl_g,
        "tl_beta": tl_beta,
        "triton_g": torch.empty(
            (1, num_batches, num_heads), device=device, dtype=torch.float32
        ),
        "triton_beta": torch.empty(
            (1, num_batches, num_heads), device=device, dtype=torch.bfloat16
        ),
    }


def make_case_runners(
    *,
    num_batches: int,
    num_heads: int,
    tilelang_kernel,
    softplus_beta: float,
    softplus_threshold: float,
    seed: int,
) -> dict[str, Any]:
    tensors = make_case_tensors(
        num_batches=num_batches,
        num_heads=num_heads,
        seed=seed,
    )

    def run_tilelang() -> None:
        tilelang_kernel(
            tensors["A_log"],
            tensors["a"],
            tensors["b"],
            tensors["dt_bias"],
            tensors["tl_g"],
            tensors["tl_beta"],
            num_batches,
            softplus_beta,
            softplus_threshold,
        )

    def run_triton() -> None:
        run_vllm_triton(
            A_log=tensors["A_log"],
            a=tensors["a"],
            b=tensors["b"],
            dt_bias=tensors["dt_bias"],
            g=tensors["triton_g"],
            beta_output=tensors["triton_beta"],
            beta=softplus_beta,
            threshold=softplus_threshold,
        )

    return {
        "tensors": tensors,
        "run_tilelang": run_tilelang,
        "run_triton": run_triton,
    }


def benchmark_case(
    *,
    num_batches: int,
    num_heads: int,
    tilelang_kernel,
    softplus_beta: float,
    softplus_threshold: float,
    warmup_iters: int,
    measure_iters: int,
    seed: int,
) -> dict[str, float | int]:
    runners = make_case_runners(
        num_batches=num_batches,
        num_heads=num_heads,
        tilelang_kernel=tilelang_kernel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        seed=seed,
    )
    tensors = runners["tensors"]
    run_tilelang = runners["run_tilelang"]
    run_triton = runners["run_triton"]

    run_tilelang()
    run_triton()
    torch.npu.synchronize()

    g_max_diff = float((tensors["tl_g"] - tensors["triton_g"][0]).abs().max().item())
    beta_max_diff = float(
        (tensors["tl_beta"].float() - tensors["triton_beta"][0].float())
        .abs()
        .max()
        .item()
    )

    tilelang_ms = measure_npu_event_ms(run_tilelang, warmup_iters, measure_iters)
    triton_ms = measure_npu_event_ms(run_triton, warmup_iters, measure_iters)
    speedup = triton_ms / tilelang_ms if tilelang_ms > 0.0 else math.nan

    return {
        "num_batches": num_batches,
        "num_heads": num_heads,
        "tilelang_ms": tilelang_ms,
        "triton_ms": triton_ms,
        "speedup": speedup,
        "g_max_diff": g_max_diff,
        "beta_max_diff": beta_max_diff,
    }


def run_worker(args: argparse.Namespace) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if len(args.num_heads_list) != 1:
        raise ValueError("--worker requires exactly one num_heads value")
    if len(args.num_batches_list) != 1:
        raise ValueError("--worker requires exactly one num_batches value")

    num_heads = args.num_heads_list[0]
    num_batches = args.num_batches_list[0]
    full_vec_core_num = detect_vec_core_num()
    launch_num_batches = (
        num_batches if args.tilelang_adaptive_launch else full_vec_core_num
    )
    tilelang_kernel = fused_gdn_gating_kernel_jit(
        num_batches=launch_num_batches,
        compile_max_batch=args.compile_max_batch,
        num_heads=num_heads,
    )

    runners = make_case_runners(
        num_batches=num_batches,
        num_heads=num_heads,
        tilelang_kernel=tilelang_kernel,
        softplus_beta=args.softplus_beta,
        softplus_threshold=args.softplus_threshold,
        seed=args.seed,
    )
    run_tilelang = runners["run_tilelang"]
    run_triton = runners["run_triton"]

    for _ in range(args.warmup_iters):
        run_tilelang()
        run_triton()
    torch.npu.synchronize()

    for _ in range(args.measure_iters):
        run_tilelang()
        run_triton()
    torch.npu.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TileLang/Triton fused_gdn_gating kernels on the Python path."
    )
    parser.add_argument(
        "--num-heads-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_NUM_HEADS_LIST),
        help="One or more head sizes to benchmark.",
    )
    parser.add_argument(
        "--num-batches-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_NUM_BATCHES_LIST),
    )
    parser.add_argument(
        "--compile-max-batch",
        type=int,
        default=DEFAULT_MAX_BATCH,
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--measure-iters", type=int, default=DEFAULT_MEASURE_ITERS)
    parser.add_argument("--softplus-beta", type=float, default=DEFAULT_SOFTPLUS_BETA)
    parser.add_argument(
        "--softplus-threshold",
        type=float,
        default=DEFAULT_SOFTPLUS_THRESHOLD,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tilelang-adaptive-launch",
        dest="tilelang_adaptive_launch",
        action="store_true",
        default=True,
        help="Adapt TileLang launch block_num by current batch size.",
    )
    parser.add_argument(
        "--no-tilelang-adaptive-launch",
        dest="tilelang_adaptive_launch",
        action="store_false",
        help="Disable adaptive launch and always use full vec_core_num as block_num.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.num_heads_list:
        raise ValueError("num_heads_list must not be empty")
    if not args.num_batches_list:
        raise ValueError("num_batches_list must not be empty")
    if any(num_heads <= 0 for num_heads in args.num_heads_list):
        raise ValueError(f"num_heads_list contains non-positive value: {args.num_heads_list}")
    if any(num_heads > DEFAULT_MAX_HEADS for num_heads in args.num_heads_list):
        raise ValueError(
            f"num_heads_list contains value larger than {DEFAULT_MAX_HEADS}: {args.num_heads_list}"
        )
    if any(num_heads not in SUPPORTED_NUM_HEADS for num_heads in args.num_heads_list):
        raise ValueError(
            "num_heads_list contains unsupported value. "
            f"Supported: {SUPPORTED_NUM_HEADS}, got {args.num_heads_list}"
        )
    if any(num_batches <= 0 for num_batches in args.num_batches_list):
        raise ValueError(
            f"num_batches_list contains non-positive value: {args.num_batches_list}"
        )
    if args.compile_max_batch <= 0:
        raise ValueError(f"compile_max_batch({args.compile_max_batch}) must be > 0")
    if args.warmup_iters < 0:
        raise ValueError(f"warmup_iters({args.warmup_iters}) must be >= 0")
    if args.measure_iters <= 0:
        raise ValueError(f"measure_iters({args.measure_iters}) must be > 0")

    max_num_batches = max(args.num_batches_list)
    if max_num_batches > args.compile_max_batch:
        raise ValueError(
            f"max(num_batches_list)={max_num_batches} exceeds compile_max_batch={args.compile_max_batch}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.worker:
        run_worker(args)
        return

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    full_vec_core_num = detect_vec_core_num()
    tilelang_kernel_cache: dict[tuple[int, int], Any] = {}

    print(
        "case,batch,num_heads,tilelang_vec_core_num,tilelang_python_ms,triton_python_ms,"
        "triton_over_tilelang,g_max_diff,beta_max_diff"
    )
    for head_idx, num_heads in enumerate(args.num_heads_list):
        for batch_idx, num_batches in enumerate(args.num_batches_list):
            launch_num_batches = (
                num_batches if args.tilelang_adaptive_launch else full_vec_core_num
            )
            block_num = select_launch_block_num(
                num_batches=launch_num_batches,
                vec_core_num=full_vec_core_num,
            )
            cache_key = (num_heads, block_num)
            if cache_key not in tilelang_kernel_cache:
                tilelang_kernel_cache[cache_key] = fused_gdn_gating_kernel_jit(
                    num_batches=launch_num_batches,
                    compile_max_batch=args.compile_max_batch,
                    num_heads=num_heads,
                )
            case_seed = args.seed + head_idx * 1000 + batch_idx
            result = benchmark_case(
                num_batches=num_batches,
                num_heads=num_heads,
                tilelang_kernel=tilelang_kernel_cache[cache_key],
                softplus_beta=args.softplus_beta,
                softplus_threshold=args.softplus_threshold,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                seed=case_seed,
            )
            print(
                f"b{num_batches}_h{num_heads},"
                f"{result['num_batches']},"
                f"{result['num_heads']},"
                f"{block_num},"
                f"{result['tilelang_ms']:.6f},"
                f"{result['triton_ms']:.6f},"
                f"{result['speedup']:.6f},"
                f"{result['g_max_diff']:.6e},"
                f"{result['beta_max_diff']:.6e}"
            )


if __name__ == "__main__":
    main()
