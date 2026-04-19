#!/bin/bash
# Collect msprof profiles for fused and naive, B=4096 and B=16384, H=32
set -e

WORKDIR="/export/home/dengyingxu1/workspace/fused_gdn_gating_opt"
SCRIPT="$WORKDIR/profile_target.py"
PROF_BASE="$WORKDIR/profiles"

collect_profile() {
    local mode=$1
    local bs=$2
    local nh=$3
    local tag="${mode}_bs${bs}_nh${nh}"
    local outdir="$PROF_BASE/$tag"

    echo "=== Profiling $tag ==="
    rm -rf "$outdir"
    mkdir -p "$outdir/msprof_raw"

    msprof --output="$outdir/msprof_raw" \
           --aic-metrics=PipeUtilization \
           --application="python3 $SCRIPT --mode $mode --batch-size $bs --num-heads $nh --warmup 10 --iters 50" \
           2>&1 | tail -5

    # Parse msprof output
    local prof_subdir=$(ls -d "$outdir/msprof_raw/PROF_"* 2>/dev/null | head -1)
    if [ -n "$prof_subdir" ]; then
        msprof --export=on --output="$prof_subdir" 2>&1 | tail -3
        echo "Profile saved: $outdir"
    else
        echo "ERROR: no PROF_ dir found in $outdir/msprof_raw"
    fi
    echo ""
}

# Collect all 4 profiles
collect_profile "fused" 4096 32
collect_profile "naive" 4096 32
collect_profile "fused" 16384 32
collect_profile "naive" 16384 32

echo "=== All profiles collected ==="
