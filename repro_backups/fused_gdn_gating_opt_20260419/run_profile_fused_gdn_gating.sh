#!/usr/bin/env bash
# One-shot msprof profiling for standalone fused_gdn_gating kernel.
# Runs inside dyx-xllm-cann85-main using attach mode.
#
# Usage:
#   BATCH_SIZE=262144 NUM_HEADS=32 TAG=baseline ./run_profile_fused_gdn_gating.sh
#   BATCH_SIZE=4096 NUM_HEADS=128 TAG=optimized ./run_profile_fused_gdn_gating.sh
#
# Outputs:
#   - Profile dir:  $WORKSPACE/fused_gdn_gating_profiles/${TAG}_bs${BS}_nh${NH}_${DATE}/
#   - Exported CSV:  op_statistic_*.csv, op_summary_*.csv
#   - Trace JSON:    msprof_*.json (open in Perfetto)
#   - Summary:       printed to stdout

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:?Set BATCH_SIZE}"
NUM_HEADS="${NUM_HEADS:-32}"
TAG="${TAG:-run}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"

CONTAINER="dyx-xllm-cann85-main"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/export/home/dengyingxu1/projects/xllm"
DATE="$(date +%Y%m%d_%H%M%S)"
PROFILE_NAME="${TAG}_bs${BATCH_SIZE}_nh${NUM_HEADS}_${DATE}"
PROFILE_DIR="${SCRIPT_DIR}/profiles/${PROFILE_NAME}"
SIGNAL_FILE="/tmp/fused_gdn_gating_msprof_ready"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_fused_gdn_gating.py"

echo "=== fused_gdn_gating msprof profiling ==="
echo "Tag:        ${TAG}"
echo "Batch:      ${BATCH_SIZE}"
echo "Heads:      ${NUM_HEADS}"
echo "Iters:      ${ITERS}"
echo "Output:     ${PROFILE_DIR}"
echo ""

# Clean up signal file
docker exec "${CONTAINER}" bash -lc "rm -f ${SIGNAL_FILE}"

# Create output directory
docker exec "${CONTAINER}" bash -lc "mkdir -p ${PROFILE_DIR}"

# Step 1: Start profiling script in background (with --wait-for-attach)
echo "[1/6] Starting kernel process (waiting for msprof attach) ..."
docker exec -d "${CONTAINER}" bash -lc "
    cd ${REPO} && \
    export PROFILING_MODE=dynamic && \
    export PYTHONPATH=\"${REPO}/xllm:${REPO}/third_party/tilelang-ascend:\${PYTHONPATH:-}\" && \
    python ${PROFILE_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-heads ${NUM_HEADS} \
        --warmup ${WARMUP} \
        --iters ${ITERS} \
        --wait-for-attach \
        --signal-file ${SIGNAL_FILE} \
        > ${PROFILE_DIR}/profile_stdout.log 2>&1
"

# Wait for process to start and print PID
sleep 3
KERNEL_PID=""
for attempt in $(seq 1 10); do
    KERNEL_PID=$(docker exec "${CONTAINER}" bash -lc "
        grep -oP 'PID=\K[0-9]+' ${PROFILE_DIR}/profile_stdout.log 2>/dev/null || true
    ")
    if [ -n "${KERNEL_PID}" ]; then
        break
    fi
    sleep 2
done

if [ -z "${KERNEL_PID}" ]; then
    echo "ERROR: Could not find kernel PID. Check ${PROFILE_DIR}/profile_stdout.log"
    docker exec "${CONTAINER}" bash -lc "cat ${PROFILE_DIR}/profile_stdout.log 2>/dev/null || true"
    exit 1
fi
echo "   Kernel PID: ${KERNEL_PID}"

# Step 2: Attach msprof
echo "[2/6] Attaching msprof ..."
docker exec "${CONTAINER}" bash -lc "
    msprof --dynamic=on \
        --pid=${KERNEL_PID} \
        --output=${PROFILE_DIR}/msprof_raw \
        --model-execution=on \
        --runtime-api=on \
        --aicpu=on \
        <<'MSPROF_CMDS'
start
MSPROF_CMDS
" &
MSPROF_BG_PID=$!

sleep 3

# Step 3: Signal the kernel to start profiled iterations
echo "[3/6] Signaling kernel to start profiled iterations ..."
docker exec "${CONTAINER}" bash -lc "touch ${SIGNAL_FILE}"

# Step 4: Wait for kernel to finish
echo "[4/6] Waiting for profiled iterations to complete ..."
for attempt in $(seq 1 120); do
    DONE=$(docker exec "${CONTAINER}" bash -lc "
        grep -c 'PROFILING DONE' ${PROFILE_DIR}/profile_stdout.log 2>/dev/null || echo 0
    ")
    if [ "${DONE}" -ge 1 ]; then
        break
    fi
    sleep 2
done

if [ "${DONE}" -lt 1 ]; then
    echo "WARNING: Kernel did not finish in time. Proceeding with msprof stop anyway."
fi

sleep 2

# Step 5: Stop msprof and export
echo "[5/6] Stopping msprof and exporting ..."
# Kill the background msprof if still running
kill "${MSPROF_BG_PID}" 2>/dev/null || true
wait "${MSPROF_BG_PID}" 2>/dev/null || true

# Find the PROF directory
PROF_SUBDIR=$(docker exec "${CONTAINER}" bash -lc "
    ls -d ${PROFILE_DIR}/msprof_raw/PROF_* 2>/dev/null | head -1 || true
")

if [ -n "${PROF_SUBDIR}" ]; then
    echo "   Exporting: ${PROF_SUBDIR}"
    docker exec "${CONTAINER}" bash -lc "
        msprof --export=on --output=${PROF_SUBDIR} 2>&1 || true
    "
else
    echo "WARNING: No PROF_* directory found under ${PROFILE_DIR}/msprof_raw/"
fi

# Step 6: Extract and display summary
echo "[6/6] Extracting summary ..."
echo ""
echo "=== Kernel stdout ==="
docker exec "${CONTAINER}" bash -lc "cat ${PROFILE_DIR}/profile_stdout.log"

echo ""
echo "=== op_statistic (fused_gdn_gating entries) ==="
docker exec "${CONTAINER}" bash -lc "
    find ${PROFILE_DIR} -name 'op_statistic_*.csv' -exec \
        grep -i 'fused_gdn_gating' {} + 2>/dev/null || echo '(none found)'
"

echo ""
echo "=== Profile artifacts ==="
docker exec "${CONTAINER}" bash -lc "
    find ${PROFILE_DIR} -name '*.csv' -o -name '*.json' | head -20
"

echo ""
echo "=== Done. Profile directory: ${PROFILE_DIR} ==="
