#!/usr/bin/env bash
# Compare two fused_gdn_gating msprof profiling runs.
#
# Usage:
#   ./compare_fused_gdn_gating.sh <baseline_dir> <optimized_dir>
#
# Example:
#   ./compare_fused_gdn_gating.sh \
#     fused_gdn_gating_profiles/baseline_bs262144_nh32_20260418/ \
#     fused_gdn_gating_profiles/optimized_bs262144_nh32_20260418/

set -euo pipefail

BASELINE_DIR="${1:?Usage: $0 <baseline_dir> <optimized_dir>}"
OPTIMIZED_DIR="${2:?Usage: $0 <baseline_dir> <optimized_dir>}"
CONTAINER="dyx-xllm-cann85-main"

echo "=== fused_gdn_gating profiling comparison ==="
echo "Baseline:  ${BASELINE_DIR}"
echo "Optimized: ${OPTIMIZED_DIR}"
echo ""

echo "--- Baseline kernel timing ---"
docker exec "${CONTAINER}" bash -lc "
    grep -i 'avg\|iters\|PROFILING' ${BASELINE_DIR}/profile_stdout.log 2>/dev/null || echo '(no stdout log)'
"
echo ""

echo "--- Optimized kernel timing ---"
docker exec "${CONTAINER}" bash -lc "
    grep -i 'avg\|iters\|PROFILING' ${OPTIMIZED_DIR}/profile_stdout.log 2>/dev/null || echo '(no stdout log)'
"
echo ""

echo "--- Baseline op_statistic (fused_gdn_gating) ---"
docker exec "${CONTAINER}" bash -lc "
    find ${BASELINE_DIR} -name 'op_statistic_*.csv' -exec head -1 {} + 2>/dev/null | head -1
    find ${BASELINE_DIR} -name 'op_statistic_*.csv' -exec \
        grep -i 'fused_gdn_gating' {} + 2>/dev/null || echo '(none)'
"
echo ""

echo "--- Optimized op_statistic (fused_gdn_gating) ---"
docker exec "${CONTAINER}" bash -lc "
    find ${OPTIMIZED_DIR} -name 'op_statistic_*.csv' -exec head -1 {} + 2>/dev/null | head -1
    find ${OPTIMIZED_DIR} -name 'op_statistic_*.csv' -exec \
        grep -i 'fused_gdn_gating' {} + 2>/dev/null || echo '(none)'
"
echo ""

echo "--- Specialization check ---"
echo "Baseline specializations:"
docker exec "${CONTAINER}" bash -lc "
    find ${BASELINE_DIR} -name 'op_statistic_*.csv' -exec \
        grep -oP 'bs\d+_nh\d+' {} + 2>/dev/null | sort -u || echo '(none found)'
"
echo ""
echo "Optimized specializations:"
docker exec "${CONTAINER}" bash -lc "
    find ${OPTIMIZED_DIR} -name 'op_statistic_*.csv' -exec \
        grep -oP 'bs\d+_nh\d+' {} + 2>/dev/null | sort -u || echo '(none found)'
"

echo ""
echo "=== Comparison complete ==="
echo "For detailed timeline analysis, open msprof_*.json files in Perfetto:"
docker exec "${CONTAINER}" bash -lc "
    echo 'Baseline:' && find ${BASELINE_DIR} -name 'msprof_*.json' 2>/dev/null | head -1
    echo 'Optimized:' && find ${OPTIMIZED_DIR} -name 'msprof_*.json' 2>/dev/null | head -1
"
