#!/usr/bin/env python3
"""Analyze msprof op_summary CSV for fused_gdn_gating kernel."""

import csv
import sys
from pathlib import Path


def analyze_op_summary(csv_path: str, label: str):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "fused_gdn_gating" in row.get("Op Name", ""):
                rows.append(row)

    if not rows:
        print(f"[{label}] No fused_gdn_gating ops found in {csv_path}")
        return

    print(f"\n{'='*80}")
    print(f"[{label}] {len(rows)} invocations of fused_gdn_gating")
    print(f"  Source: {csv_path}")
    print(f"{'='*80}")

    # Parse numeric fields
    def parse_us(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Key fields to extract
    fields = [
        "Task Duration(us)",
        "Task Wait Time(us)",
        "aiv_vec_time(us)",
        "aiv_mte2_time(us)",
        "aiv_mte3_time(us)",
        "aiv_scalar_time(us)",
        "aiv_vec_ratio(%)",
        "aiv_mte2_ratio(%)",
        "aiv_mte3_ratio(%)",
    ]

    # Print header
    print(f"\n{'Inv':>4} ", end="")
    short_names = {
        "Task Duration(us)": "Duration",
        "Task Wait Time(us)": "WaitTime",
        "aiv_vec_time(us)": "VEC",
        "aiv_mte2_time(us)": "MTE2",
        "aiv_mte3_time(us)": "MTE3",
        "aiv_scalar_time(us)": "SCALAR",
        "aiv_vec_ratio(%)": "VEC%",
        "aiv_mte2_ratio(%)": "MTE2%",
        "aiv_mte3_ratio(%)": "MTE3%",
    }
    for f in fields:
        print(f"{short_names[f]:>10}", end="")
    print()
    print("-" * (4 + 10 * len(fields) + 1))

    # Print each invocation
    warm_durations = []
    warm_waits = []
    warm_vec = []
    warm_mte2 = []
    warm_mte3 = []
    warm_scalar = []

    for i, row in enumerate(rows):
        print(f"{i:>4} ", end="")
        for f in fields:
            val = row.get(f, "N/A")
            print(f"{val:>10}", end="")
        print()

        # Collect warm stats (skip first invocation = cold)
        if i > 0:
            d = parse_us(row.get("Task Duration(us)"))
            w = parse_us(row.get("Task Wait Time(us)"))
            v = parse_us(row.get("aiv_vec_time(us)"))
            m2 = parse_us(row.get("aiv_mte2_time(us)"))
            m3 = parse_us(row.get("aiv_mte3_time(us)"))
            sc = parse_us(row.get("aiv_scalar_time(us)"))
            if d is not None:
                warm_durations.append(d)
            if w is not None:
                warm_waits.append(w)
            if v is not None:
                warm_vec.append(v)
            if m2 is not None:
                warm_mte2.append(m2)
            if m3 is not None:
                warm_mte3.append(m3)
            if sc is not None:
                warm_scalar.append(sc)

    # Summary stats for warm invocations
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def mn(lst):
        return min(lst) if lst else 0

    def mx(lst):
        return max(lst) if lst else 0

    n_warm = len(warm_durations)
    print(f"\n--- Warm invocation stats (n={n_warm}, excluding 1st cold) ---")
    print(f"  Task Duration : avg={avg(warm_durations):.1f}us  min={mn(warm_durations):.1f}us  max={mx(warm_durations):.1f}us")
    print(f"  Task WaitTime : avg={avg(warm_waits):.1f}us  min={mn(warm_waits):.1f}us  max={mx(warm_waits):.1f}us")
    print(f"  aiv_vec       : avg={avg(warm_vec):.1f}us  min={mn(warm_vec):.1f}us  max={mx(warm_vec):.1f}us")
    print(f"  aiv_mte2      : avg={avg(warm_mte2):.1f}us  min={mn(warm_mte2):.1f}us  max={mx(warm_mte2):.1f}us")
    print(f"  aiv_mte3      : avg={avg(warm_mte3):.1f}us  min={mn(warm_mte3):.1f}us  max={mx(warm_mte3):.1f}us")
    print(f"  aiv_scalar    : avg={avg(warm_scalar):.1f}us  min={mn(warm_scalar):.1f}us  max={mx(warm_scalar):.1f}us")

    total_device = avg(warm_durations)
    total_host_wait = avg(warm_waits)
    total_e2e = total_device + total_host_wait
    print(f"\n--- Time breakdown (warm avg) ---")
    print(f"  Host wait (dispatch overhead) : {total_host_wait:.1f}us ({total_host_wait/total_e2e*100:.1f}%)")
    print(f"  Device execution              : {total_device:.1f}us ({total_device/total_e2e*100:.1f}%)")
    print(f"  E2E per invocation            : {total_e2e:.1f}us")
    print(f"    VEC compute                 : {avg(warm_vec):.1f}us")
    print(f"    MTE2 (GM->UB DMA)           : {avg(warm_mte2):.1f}us")
    print(f"    MTE3 (UB->GM DMA)           : {avg(warm_mte3):.1f}us")
    print(f"    Scalar                      : {avg(warm_scalar):.1f}us")


def analyze_op_statistic(csv_path: str, label: str):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "fused_gdn_gating" in row.get("OP Type", ""):
                rows.append(row)

    if not rows:
        return

    print(f"\n--- op_statistic for {label} ---")
    for row in rows:
        for k, v in row.items():
            if v and v != "N/A":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    profiles = {
        "B=16384 H=32": {
            "op_summary": "/export/home/dengyingxu1/workspace/fused_gdn_gating_opt/profiles/phase3_bs16384_nh32/msprof_raw/PROF_000001_20260418184001126_IDEOREOILDGKDQOA/mindstudio_profiler_output/op_summary_20260418184008.csv",
            "op_statistic": "/export/home/dengyingxu1/workspace/fused_gdn_gating_opt/profiles/phase3_bs16384_nh32/msprof_raw/PROF_000001_20260418184001126_IDEOREOILDGKDQOA/mindstudio_profiler_output/op_statistic_20260418184008.csv",
        },
        "B=4096 H=32": {
            "op_summary": "/export/home/dengyingxu1/workspace/fused_gdn_gating_opt/profiles/phase3_bs4096_nh32/msprof_raw/PROF_000001_20260418184028848_GGBHJJMJLIAOEENB/mindstudio_profiler_output/op_summary_20260418184036.csv",
            "op_statistic": "/export/home/dengyingxu1/workspace/fused_gdn_gating_opt/profiles/phase3_bs4096_nh32/msprof_raw/PROF_000001_20260418184028848_GGBHJJMJLIAOEENB/mindstudio_profiler_output/op_statistic_20260418184036.csv",
        },
    }

    for label, paths in profiles.items():
        analyze_op_summary(paths["op_summary"], label)
        analyze_op_statistic(paths["op_statistic"], label)
