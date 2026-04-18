#!/usr/bin/env bash
# Run fused_gdn_gating TileLang kernel: ref check + msprof benchmark.
#
# Usage (from host or inside dyx-xllm-cann85-main container):
#   bash xllm/compiler/tilelang/targets/ascend/kernels/run_fused_gdn_gating.sh
#
# The script automatically enters the NPU container when run on the host.
set -euo pipefail

# ---------------------------------------------------------------------------
# Auto-enter NPU container when running on the host
# ---------------------------------------------------------------------------
NPU_CONTAINER="dyx-xllm-cann85-main"
REPO_HOST="/export/home/dengyingxu1/projects/xllm"
SCRIPT_REL="xllm/compiler/tilelang/targets/ascend/kernels/run_fused_gdn_gating.sh"

if [[ -z "${_INSIDE_NPU_CONTAINER:-}" ]]; then
    exec docker exec "$NPU_CONTAINER" /bin/bash -lc \
        "export _INSIDE_NPU_CONTAINER=1; cd $REPO_HOST && bash $SCRIPT_REL $*"
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
XLLM_PKG="$REPO_ROOT/xllm"
TL_ROOT="$REPO_ROOT/third_party/tilelang-ascend"
KERNELS_DIR="$XLLM_PKG/compiler/tilelang/targets/ascend/kernels"
BENCH_DIR="$KERNELS_DIR/benchmarks"
TMP_DIR="$REPO_ROOT/.tmp"
mkdir -p "$TMP_DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export TL_ROOT
export PYTHONPATH="$XLLM_PKG${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

if [[ -f "$TL_ROOT/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$TL_ROOT/set_env.sh"
fi

MODULE="compiler.tilelang.targets.ascend.kernels.fused_gdn_gating"
BENCH_SCRIPT="fused_gdn_gating_python_perf_compare.py"

echo "=== fused_gdn_gating TileLang test ==="
echo "REPO_ROOT           : $REPO_ROOT"
echo "XLLM_PKG            : $XLLM_PKG"
echo "TL_ROOT             : $TL_ROOT"
echo "MODULE              : $MODULE"
echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo ""

cd "$XLLM_PKG"

# ---------------------------------------------------------------------------
# 1. Ref check (correctness — must pass before any performance testing)
# ---------------------------------------------------------------------------
echo "=== [1/4] Ref check (all SUPPORTED_NUM_HEADS) ==="
NUM_HEADS_LIST=(4 6 8 12 16 24 32 48 64 128)
for nh in "${NUM_HEADS_LIST[@]}"; do
    echo -n "  num_heads=$nh num_batches=16 ... "
    python -m "$MODULE" \
        --output "$TMP_DIR/fused_gdn_gating_nh${nh}.cpp" \
        --num-heads "$nh" \
        --batch-size 48 \
        --ref-num-batches 16 \
        --ref-num-heads-list "$nh"
done
echo ""

# ---------------------------------------------------------------------------
# 2. Codegen inspection
# ---------------------------------------------------------------------------
echo "=== [2/4] Codegen stats (num_heads=32, batch_size=48) ==="
CPP="$TMP_DIR/fused_gdn_gating_nh32.cpp"
if [[ -f "$CPP" ]]; then
    echo "  Lines          : $(wc -l < "$CPP")"
    echo "  PipeBarrier<V> : $(grep -c 'PipeBarrier<PIPE_V>' "$CPP" || true)"
    echo "  PipeBarrier<ALL>: $(grep -c 'PipeBarrier<PIPE_ALL>' "$CPP" || true)"
    echo "  PipeBarrier<MTE2>: $(grep -c 'PipeBarrier<PIPE_MTE2>' "$CPP" || true)"
    echo "  PipeBarrier<MTE3>: $(grep -c 'PipeBarrier<PIPE_MTE3>' "$CPP" || true)"
    echo "  SetFlag        : $(grep -c 'SetFlag' "$CPP" || true)"
    echo "  WaitFlag       : $(grep -c 'WaitFlag' "$CPP" || true)"
    echo "  DataCopyPad    : $(grep -c 'DataCopyPad' "$CPP" || true)"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Python perf compare (TileLang vs Triton, Python-side timing + accuracy)
# ---------------------------------------------------------------------------
echo "=== [3/4] Python perf compare (TileLang vs Triton) ==="
python "$BENCH_DIR/$BENCH_SCRIPT" \
    --num-heads-list 32 128 \
    --num-batches-list 1 2 4 8 16 32 48 1024 4096 16384 65536 262144 \
    --compile-max-batch 262144 \
    --warmup-iters 20 \
    --measure-iters 200 \
    2>&1 | tee "$TMP_DIR/fused_gdn_gating_perf_$(date +%Y%m%d_%H%M%S).csv"

# ---------------------------------------------------------------------------
# 4. msprof performance benchmark (device-level kernel task duration)
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/4] msprof performance benchmark ==="
python "$BENCH_DIR/kernel_msprof_task_duration.py" \
    --runner-script "$BENCH_DIR/$BENCH_SCRIPT" \
    --worker-cmd-template \
    "{python} {runner_script} --worker --num-heads-list {num_heads} --num-batches-list {num_batches} --compile-max-batch {compile_max_batch} --warmup-iters {warmup_iters} --measure-iters {measure_iters} --softplus-beta {softplus_beta} --softplus-threshold {softplus_threshold} --seed {seed}" \
    --num-batches-list 1 2 4 8 16 32 48 1024 4096 16384 65536 262144 \
    --num-heads-list 32 128 \
    --compile-max-batch 262144 \
    --warmup-iters 20 \
    --measure-iters 100 \
    --kernel tilelang=fused_gdn_gating_kernel_kernel \
    --kernel triton=vllm_fused_gdn_gating_kernel \
    --kernel-match-mode exact \
    --baseline-label tilelang \
    2>&1 | tee "$TMP_DIR/fused_gdn_gating_msprof_$(date +%Y%m%d_%H%M%S).csv"

echo ""
echo "=== Done. Results saved to $TMP_DIR/ ==="
