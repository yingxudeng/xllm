#!/usr/bin/env python3

"""
Run msprof on a Python worker script and extract per-kernel task duration from
msprof trace JSON into CSV output.

This script is generic enough to benchmark multiple kernels by:
  1) configuring the worker command template, and
  2) listing one or more kernel label/pattern pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[6]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

DEFAULT_MSPROF_OUTPUT_ROOT = "/tmp/msprof_task_duration"
DEFAULT_RUNNER_SCRIPT = Path(__file__).resolve().with_name(
    "fused_gdn_gating_python_perf_compare.py"
)
DEFAULT_WORKER_CMD_TEMPLATE = (
    "{python} {runner_script} "
    "--worker "
    "--num-heads-list {num_heads} "
    "--num-batches-list {num_batches} "
    "--compile-max-batch {compile_max_batch} "
    "--warmup-iters {warmup_iters} "
    "--measure-iters {measure_iters} "
    "--softplus-beta {softplus_beta} "
    "--softplus-threshold {softplus_threshold} "
    "--seed {seed}"
)

# Keep defaults compatible with current fused_gdn_gating scripts.
DEFAULT_NUM_BATCHES_LIST = (16, 48, 1024, 4096, 16384, 65536, 262144)
DEFAULT_NUM_HEADS_LIST = (32, 128)
DEFAULT_COMPILE_MAX_BATCH = 262144
DEFAULT_WARMUP_ITERS = 20
DEFAULT_MEASURE_ITERS = 200
DEFAULT_SOFTPLUS_BETA = 1.0
DEFAULT_SOFTPLUS_THRESHOLD = 20.0
DEFAULT_SEED = 42

DEFAULT_TILELANG_KERNEL_NAME = "fused_gdn_gating_kernel_kernel"
DEFAULT_TRITON_KERNEL_NAME = "vllm_fused_gdn_gating_kernel"


def _find_latest_profile_json(profile_dir: Path) -> Path:
    matches = sorted(profile_dir.rglob("msprof_*.json"))
    if not matches:
        raise FileNotFoundError(f"No msprof_*.json found under {profile_dir}")
    return matches[-1]


def _load_trace_events(profile_dir: Path) -> list[dict[str, Any]]:
    json_path = _find_latest_profile_json(profile_dir)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON event list in {json_path}")
    return data


def _event_name_matches(name: str, pattern: str, match_mode: str) -> bool:
    if match_mode == "exact":
        return name == pattern
    if match_mode == "contains":
        return pattern in name
    if match_mode == "regex":
        return re.search(pattern, name) is not None
    raise ValueError(f"Unsupported kernel_match_mode={match_mode!r}")


def _extract_kernel_task_duration_from_events(
    events: list[dict[str, Any]],
    *,
    kernel_pattern: str,
    sample_count: int,
    match_mode: str,
) -> dict[str, float | int | str]:
    matched = [
        event
        for event in events
        if event.get("ph") == "X"
        and _event_name_matches(str(event.get("name", "")), kernel_pattern, match_mode)
    ]
    if not matched:
        raise ValueError(
            f"Kernel pattern {kernel_pattern!r} was not found in trace "
            f"(match_mode={match_mode})"
        )

    durations = [float(event["dur"]) for event in matched]
    tail = durations[-sample_count:] if sample_count > 0 else durations
    task_type = str(matched[-1].get("args", {}).get("Task Type", ""))

    return {
        "task_type": task_type,
        "count_total": len(durations),
        "avg_task_us": float(sum(tail) / len(tail)),
        "min_task_us": float(min(tail)),
        "max_task_us": float(max(tail)),
    }


def _parse_kernel_specs(
    *,
    kernel_specs: list[str],
    fallback_tilelang_name: str,
    fallback_triton_name: str,
) -> list[tuple[str, str]]:
    if not kernel_specs:
        return [
            ("tilelang", fallback_tilelang_name),
            ("triton", fallback_triton_name),
        ]

    parsed: list[tuple[str, str]] = []
    labels_seen: set[str] = set()
    for spec in kernel_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --kernel {spec!r}; expected format LABEL=PATTERN"
            )
        label, pattern = spec.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label:
            raise ValueError(f"Invalid --kernel {spec!r}; label is empty")
        if not pattern:
            raise ValueError(f"Invalid --kernel {spec!r}; pattern is empty")
        if label in labels_seen:
            raise ValueError(f"Duplicate kernel label in --kernel: {label!r}")
        labels_seen.add(label)
        parsed.append((label, pattern))
    return parsed


def _build_worker_cmd(
    args: argparse.Namespace,
    *,
    runner_script_path: Path,
    num_batches: int,
    num_heads: int,
    seed: int,
) -> list[str]:
    template_ctx = {
        "python": sys.executable,
        "runner_script": str(runner_script_path),
        "num_batches": num_batches,
        "num_heads": num_heads,
        "compile_max_batch": args.compile_max_batch,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "softplus_beta": args.softplus_beta,
        "softplus_threshold": args.softplus_threshold,
        "seed": seed,
    }

    try:
        rendered = args.worker_cmd_template.format(**template_ctx)
    except KeyError as exc:
        raise ValueError(
            "Unknown placeholder in --worker-cmd-template: "
            f"{exc.args[0]!r}. Supported keys: {sorted(template_ctx.keys())}"
        ) from exc

    worker_cmd = shlex.split(rendered)
    for extra_arg in args.worker_arg:
        worker_cmd.extend(shlex.split(extra_arg))
    if not worker_cmd:
        raise ValueError("Rendered worker command is empty")
    return worker_cmd


def profile_case_with_msprof(
    *,
    runner_script_path: Path,
    num_batches: int,
    num_heads: int,
    seed: int,
    msprof_output_root: Path,
    msprof_bin: str,
    keep_msprof_output: bool,
    kernel_specs: list[tuple[str, str]],
    match_mode: str,
    baseline_label: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    case_name = f"h{num_heads}_b{num_batches}"
    msprof_case_dir = msprof_output_root / case_name
    if msprof_case_dir.exists():
        shutil.rmtree(msprof_case_dir)
    msprof_case_dir.mkdir(parents=True, exist_ok=True)

    worker_cmd = _build_worker_cmd(
        args,
        runner_script_path=runner_script_path,
        num_batches=num_batches,
        num_heads=num_heads,
        seed=seed,
    )
    worker_cmd_str = " ".join(shlex.quote(arg) for arg in worker_cmd)

    msprof_cmd = [
        msprof_bin,
        f"--output={msprof_case_dir}",
        f"--application={worker_cmd_str}",
    ]
    subprocess.run(msprof_cmd, check=True)

    events = _load_trace_events(msprof_case_dir)

    result: dict[str, float | int | str] = {
        "case": f"b{num_batches}_h{num_heads}",
        "batch": num_batches,
        "num_heads": num_heads,
        "profile_dir": str(msprof_case_dir),
    }

    per_kernel_stats: dict[str, dict[str, float | int | str]] = {}
    for label, pattern in kernel_specs:
        stats = _extract_kernel_task_duration_from_events(
            events,
            kernel_pattern=pattern,
            sample_count=args.measure_iters,
            match_mode=match_mode,
        )
        per_kernel_stats[label] = stats
        result[f"{label}_task_type"] = str(stats["task_type"])
        result[f"{label}_avg_task_us"] = float(stats["avg_task_us"])
        result[f"{label}_min_task_us"] = float(stats["min_task_us"])
        result[f"{label}_max_task_us"] = float(stats["max_task_us"])
        result[f"{label}_task_count"] = int(stats["count_total"])

    baseline_avg = float(per_kernel_stats[baseline_label]["avg_task_us"])
    for label, _ in kernel_specs:
        if label == baseline_label:
            continue
        cur_avg = float(per_kernel_stats[label]["avg_task_us"])
        result[f"{label}_over_{baseline_label}"] = (
            cur_avg / baseline_avg if baseline_avg > 0.0 else math.nan
        )

    if not keep_msprof_output:
        shutil.rmtree(msprof_case_dir)

    return result


def _build_fieldnames(
    *,
    kernel_specs: list[tuple[str, str]],
    baseline_label: str,
    include_profile_dir_column: bool,
) -> list[str]:
    labels = [label for label, _ in kernel_specs]

    fieldnames = ["case", "batch", "num_heads"]
    fieldnames.extend([f"{label}_task_type" for label in labels])
    fieldnames.extend([f"{label}_avg_task_us" for label in labels])
    fieldnames.extend(
        [f"{label}_over_{baseline_label}" for label in labels if label != baseline_label]
    )
    fieldnames.extend([f"{label}_min_task_us" for label in labels])
    fieldnames.extend([f"{label}_max_task_us" for label in labels])
    fieldnames.extend([f"{label}_task_count" for label in labels])
    if include_profile_dir_column:
        fieldnames.append("profile_dir")
    return fieldnames


def _write_csv(
    rows: list[dict[str, object]],
    csv_output: Path,
    *,
    fieldnames: list[str],
) -> None:
    if not rows:
        return
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_csv_value(v: object) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.6f}"
    return str(v)


def _format_csv_row(row: dict[str, object], *, fieldnames: list[str]) -> str:
    return ",".join(_format_csv_value(row.get(name, "")) for name in fieldnames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run msprof on a Python worker command and extract per-kernel "
            "task duration from trace JSON."
        )
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=DEFAULT_RUNNER_SCRIPT,
        help="Worker script path used by --worker-cmd-template.",
    )
    parser.add_argument(
        "--worker-cmd-template",
        default=DEFAULT_WORKER_CMD_TEMPLATE,
        help=(
            "Worker command template. Supports placeholders: "
            "{python}, {runner_script}, {num_heads}, {num_batches}, "
            "{compile_max_batch}, {warmup_iters}, {measure_iters}, "
            "{softplus_beta}, {softplus_threshold}, {seed}."
        ),
    )
    parser.add_argument(
        "--worker-arg",
        action="append",
        default=[],
        help=(
            "Extra worker arg fragment appended after template rendering. "
            "Can be repeated, e.g. --worker-arg \"--dtype bf16\"."
        ),
    )

    parser.add_argument(
        "--num-heads-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_NUM_HEADS_LIST),
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
        default=DEFAULT_COMPILE_MAX_BATCH,
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
        "--kernel",
        action="append",
        default=[],
        help=(
            "Kernel label and matching pattern. Repeatable. "
            "Format: LABEL=PATTERN, e.g. --kernel tilelang=my_kernel."
        ),
    )
    parser.add_argument(
        "--kernel-match-mode",
        choices=("exact", "contains", "regex"),
        default="exact",
        help="How PATTERN in --kernel matches msprof event name.",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Baseline label used for ratio columns <label>_over_<baseline>.",
    )
    parser.add_argument(
        "--tilelang-kernel-name",
        default=DEFAULT_TILELANG_KERNEL_NAME,
        help=(
            "Backward-compatible fallback for tilelang kernel event name "
            "(used when --kernel is not specified)."
        ),
    )
    parser.add_argument(
        "--triton-kernel-name",
        default=DEFAULT_TRITON_KERNEL_NAME,
        help=(
            "Backward-compatible fallback for triton kernel event name "
            "(used when --kernel is not specified)."
        ),
    )

    parser.add_argument(
        "--msprof-output-root",
        type=Path,
        default=Path(DEFAULT_MSPROF_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--msprof-bin",
        default="msprof",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, only stdout CSV is printed.",
    )
    parser.add_argument(
        "--keep-msprof-output",
        action="store_true",
        help="Keep raw msprof output directories for each case.",
    )
    parser.add_argument(
        "--include-profile-dir-column",
        action="store_true",
        help="Include profile_dir column in stdout/csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved worker command for the first case and exit.",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    *,
    kernel_specs: list[tuple[str, str]],
    baseline_label: str,
) -> None:
    if not args.num_heads_list:
        raise ValueError("num_heads_list must not be empty")
    if not args.num_batches_list:
        raise ValueError("num_batches_list must not be empty")
    if any(num_heads <= 0 for num_heads in args.num_heads_list):
        raise ValueError(f"num_heads_list contains non-positive value: {args.num_heads_list}")
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
            f"max(num_batches_list)={max_num_batches} exceeds "
            f"compile_max_batch={args.compile_max_batch}"
        )

    if not args.runner_script.exists():
        raise FileNotFoundError(f"runner_script does not exist: {args.runner_script}")

    labels = {label for label, _ in kernel_specs}
    if baseline_label not in labels:
        raise ValueError(
            f"baseline_label {baseline_label!r} is not in kernel labels: {sorted(labels)}"
        )

    if args.kernel_match_mode == "regex":
        for _, pattern in kernel_specs:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex kernel pattern {pattern!r}: {exc}") from exc


def main() -> None:
    args = parse_args()
    kernel_specs = _parse_kernel_specs(
        kernel_specs=args.kernel,
        fallback_tilelang_name=args.tilelang_kernel_name,
        fallback_triton_name=args.triton_kernel_name,
    )
    baseline_label = args.baseline_label or kernel_specs[0][0]
    validate_args(args, kernel_specs=kernel_specs, baseline_label=baseline_label)

    runner_script_path = args.runner_script.resolve()
    args.msprof_output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview_cmd = _build_worker_cmd(
            args,
            runner_script_path=runner_script_path,
            num_batches=args.num_batches_list[0],
            num_heads=args.num_heads_list[0],
            seed=args.seed,
        )
        print(" ".join(shlex.quote(arg) for arg in preview_cmd))
        return

    fieldnames = _build_fieldnames(
        kernel_specs=kernel_specs,
        baseline_label=baseline_label,
        include_profile_dir_column=args.include_profile_dir_column,
    )

    rows: list[dict[str, object]] = []
    print(",".join(fieldnames))
    for head_idx, num_heads in enumerate(args.num_heads_list):
        for batch_idx, num_batches in enumerate(args.num_batches_list):
            case_seed = args.seed + head_idx * 1000 + batch_idx
            result = profile_case_with_msprof(
                runner_script_path=runner_script_path,
                num_batches=num_batches,
                num_heads=num_heads,
                seed=case_seed,
                msprof_output_root=args.msprof_output_root,
                msprof_bin=args.msprof_bin,
                keep_msprof_output=args.keep_msprof_output,
                kernel_specs=kernel_specs,
                match_mode=args.kernel_match_mode,
                baseline_label=baseline_label,
                args=args,
            )
            rows.append(result)
            print(_format_csv_row(result, fieldnames=fieldnames))

    if args.csv_output is not None:
        _write_csv(rows, args.csv_output, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
