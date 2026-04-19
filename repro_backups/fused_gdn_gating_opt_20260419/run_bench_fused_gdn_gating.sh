#!/usr/bin/env bash
# Run fused_gdn_gating benchmark inside dyx-xllm-cann85-main container.
# Results are saved to a timestamped CSV under workspace/.
#
# Usage:
#   ./run_bench_fused_gdn_gating.sh [extra args for bench script]
#
# Examples:
#   ./run_bench_fused_gdn_gating.sh
#   ./run_bench_fused_gdn_gating.sh --batch-sizes 4096 262144 --num-heads-list 32

set -euo pipefail

CONTAINER="dyx-xllm-cann85-main"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/bench_fused_gdn_gating.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_CSV="${SCRIPT_DIR}/results/bench_${TIMESTAMP}.csv"

echo "=== fused_gdn_gating benchmark ==="
echo "Container: ${CONTAINER}"
echo "Output:    ${OUTPUT_CSV}"
echo ""

docker exec "${CONTAINER}" bash -lc "
  mkdir -p ${SCRIPT_DIR}/results && \
  cd /export/home/dengyingxu1/projects/xllm && \
  python ${BENCH_SCRIPT} --output-csv ${OUTPUT_CSV} $*
"

echo ""
echo "=== Done. Results at: ${OUTPUT_CSV} ==="
