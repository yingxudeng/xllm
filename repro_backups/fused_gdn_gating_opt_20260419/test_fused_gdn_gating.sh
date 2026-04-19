#!/usr/bin/env bash
# Run fused_gdn_gating correctness tests inside dyx-xllm-cann85-main.
#
# Steps:
#   1. Python JIT reference check (multiple num_heads)
#   2. C++ wrapper test (GoogleTest via setup.py)
#
# Usage:
#   ./test_fused_gdn_gating.sh          # run both Python + C++ tests
#   ./test_fused_gdn_gating.sh python    # Python ref check only
#   ./test_fused_gdn_gating.sh cpp       # C++ wrapper test only

set -euo pipefail

CONTAINER="dyx-xllm-cann85-main"
REPO="/export/home/dengyingxu1/projects/xllm"
MODE="${1:-all}"

run_python_test() {
    echo "=== Python JIT reference check ==="
    docker exec "${CONTAINER}" bash -lc "
        cd ${REPO} && \
        python -c \"
import sys
sys.path.insert(0, '.')
from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import _run_ref_suite

batch_sizes = [1, 4, 17, 29, 48, 131, 257]
heads_list = [8, 16, 32, 48, 64, 128]

for bs in batch_sizes:
    print(f'--- batch_size={bs} ---')
    _run_ref_suite(
        num_batches=bs,
        compile_max_batch=max(bs, 4096),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        ref_num_heads_list=heads_list,
    )
    print()

print('ALL PYTHON REF CHECKS PASSED')
\"
    "
}

run_cpp_test() {
    echo "=== C++ wrapper test ==="
    docker exec "${CONTAINER}" bash -lc "
        cd ${REPO} && \
        python setup.py build_test --test-name TileLangFusedGdnGatingWrapperTest
    "
}

case "${MODE}" in
    python)
        run_python_test
        ;;
    cpp)
        run_cpp_test
        ;;
    all)
        run_python_test
        echo ""
        run_cpp_test
        ;;
    *)
        echo "Usage: $0 [all|python|cpp]"
        exit 1
        ;;
esac

echo ""
echo "=== All requested tests passed ==="
