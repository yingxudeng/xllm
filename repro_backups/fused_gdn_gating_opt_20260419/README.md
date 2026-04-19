# fused_gdn_gating 算子性能优化

## Git 备份说明

本目录是从以下工作目录整理出的轻量 Git 备份，方便后续继续工作：

- `/export/home/dengyingxu1/workspace/fused_gdn_gating_opt`

为避免将大量 profiling 原始产物提交进仓库，本备份**未包含**原目录下的 `profiles/msprof_raw/` 等重型数据。
当前保留的内容是：

- 所有 Python / Shell 工作脚本
- `README.md`
- `results/*.csv`

如需恢复或继续分析原始 profiling 结果，请回到原始工作目录查看完整 `profiles/` 内容。

## 目标

优化 TileLang Ascend 算子 `fused_gdn_gating`，使 batch_size 1~262144 全范围性能良好。

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0: 脚本体系 | DONE | benchmark / profiling / test / compare 脚本就绪 |
| Phase 1: 多行处理 | DONE | 每次迭代处理 R 行（R=16 for heads≤64, R=8 for heads=128），方案 B（复制广播 buffer） |
| Phase 2: 增大 compile_max_batch | DONE | 4096 → 262144，单次 kernel launch 覆盖全量 batch |
| Phase 3: 消除 DMA 瓶颈 | DONE | multi-row bulk DMA + 32-row chunks，详见下方 |
| Phase 3.5: msprof 分析 | DONE | 发现 JIT ~137us 开销来自 Cython wrapper，非 kernel dispatch；direct call 14.2us vs naive 66.7us |
| Phase 3.6: C++ AOT 验证 | DONE | 修复 bulk DMA 余数 chunk 越界 bug，C++ AOT 全 12 case 通过，benchmark 确认 2.5-2.9x faster |
| Phase 3.7: 小 batch 优化 | DONE | 修复 B≤rows_per_iter 时 task 利用率崩塌，TileLang 全 batch 范围优于 Triton |
| Phase 4: 双缓冲 | DONE | codegen 已生成双缓冲 flag 模式，但 msprof 实测无收益，H=32 反而慢 2-12%，当前不采纳 |

## 关键文件

| 文件 | 作用 |
|------|------|
| `xllm/compiler/tilelang/targets/ascend/kernels/fused_gdn_gating.py` | kernel 实现（主要修改文件） |
| `xllm/core/kernels/npu/tilelang/fused_gdn_gating_wrapper.cpp` | C++ wrapper（改 kCompileMaxBatch） |
| `xllm/core/kernels/npu/tilelang/fused_gdn_gating_wrapper_test.cpp` | C++ 正确性测试 + benchmark |
| `xllm/compiler/tilelang/targets/ascend/kernels/utils.py` | pass configs, detect_vec_core_num |

## 优化思路

### Phase 1: 多行处理

**问题**: 当前每次循环只处理 1 行（1, dim），DMA 传输量小、循环开销大。

**方案**: 每次处理 R 行 (R, dim)。方案 B — 把 `dt_bias_ub` 和 `neg_exp_A_ub` 也复制成 (R, dim)，所有 tile ops 统一操作，无需广播子循环。

**R 值选择** (UB 64KB 预算):

| num_heads | ub_tensor_dim | 每行 UB 开销 | 选用 R |
|-----------|--------------|-------------|--------|
| 4-64      | 64           | ~2120B      | 16     |
| 128       | 128          | ~4240B      | 8      |

**改动要点**:
1. 新增 `_compute_rows_per_iter(num_heads)` 函数
2. 所有 per-iteration buffer 从 `(1, dim)` 改为 `(R, dim)`
3. `dt_bias_ub`, `neg_exp_A_ub` 也分配 `(R, dim)`，循环外从 1 行复制到 R 行
4. 主循环：`for iter_idx in T.serial(num_full_iters)` 每次处理 R 行
5. 余数循环：`for rem_idx in T.serial(remainder)` 逐行处理剩余

### Phase 2: 增大 compile_max_batch

**问题**: `DEFAULT_MAX_BATCH=4096`，C++ wrapper 分 chunk 调用。batch=262144 时 64 次 kernel launch。

**方案**: 改为 262144。compile_max_batch 只影响 tensor 声明形状，不影响 UB。

### Phase 3: 消除 DMA 瓶颈（已完成）

**问题**: msprof 显示 DMA 占 97.6%。每个 chunk 64 次 per-row DMA（16行×4tensor），因 GM stride ≠ UB stride。

**方案**: 当 `ub_tensor_dim = num_heads` 时 GM/UB stride 匹配 → DataCopyPad 生成零间隙 burst DMA。

**关键改动**:
1. `_can_use_bulk_dma()` — 检查 num_heads % 16 == 0（bf16 DataCopy 块对齐）
2. bulk DMA 路径: `T.copy(a[base_row, 0], a_half_ub)` → `copy_gm_to_ub<bf16, N, M>` (multi-row)
3. `_compute_rows_per_iter` 支持 ub_dim 参数 + 32-row 选项
4. mask 计算修正: ceiling division `(ub_tensor_dim + 7) // 8`
5. C++ wrapper: `CHECK_EQ(a.stride(0), a.size(1))` row-contiguity 检查

**Bulk DMA 适用**: H=16,32,48,64,128 (num_heads 是 16 的倍数)
**Fallback per-row**: H=4,6,8,12,24

**H=8/12/24 bulk DMA 失败根因**: DataCopyPad 要求行宽是 block 粒度倍数。bf16 block=16 元素, 8%16≠0, 12%16≠0, 24%16≠0。

### Phase 3: 手动 DMA/计算重叠（原 Phase 3，待定）

copy(b) 可与 softplus 计算(只依赖 a)并行。用 `T.set_flag/T.wait_flag` 手动控制。

### Phase 4: 双缓冲（已验证，不采纳）

目标是通过 `T.Pipelined(num_stages=2)` 预取下一批数据，让 GM→UB DMA 与当前 chunk 计算重叠。

**代码生成验证**:

- `SetFlag=7`
- `WaitFlag=7`

说明生成代码中已经出现预期的双缓冲 flag 同步模式，问题不在于 codegen 没生效，而在于运行时收益不足以覆盖同步开销。

## 脚本使用

所有脚本在 `dyx-xllm-cann85-main` 容器中执行。

```bash
# 正确性测试
./test_fused_gdn_gating.sh              # Python + C++
./test_fused_gdn_gating.sh python        # 仅 Python ref check

# 性能 benchmark
./run_bench_fused_gdn_gating.sh          # 全量 batch sizes
./run_bench_fused_gdn_gating.sh --batch-sizes 4096 16384 --num-heads-list 32

# msprof profiling
BATCH_SIZE=16384 NUM_HEADS=32 TAG=baseline ./run_profile_fused_gdn_gating.sh
BATCH_SIZE=16384 NUM_HEADS=32 TAG=optimized ./run_profile_fused_gdn_gating.sh

# 对比两次 profiling
./compare_fused_gdn_gating.sh profiles/baseline_*/ profiles/optimized_*/
```

输出位置:
- `results/` — benchmark CSV
- `profiles/` — msprof 输出 (op_statistic, op_summary, msprof_*.json)

## 硬件参数

- Ascend A2/A3, 48 vector cores, VEC_NUM=2
- UB: 64KB per vector core
- 向量处理: 256B/iter (float32 = 64 元素)

## 验证循环

每次修改后**必须**按顺序执行，**精度验证必须先于性能测试**：

```
修改代码 → 精度验证 (必须全部通过) → bench (延迟) → profile (算子级) → compare (前后对比)
```

### 精度验证（强制，不可跳过）

**原则**: 任何对 kernel 代码的修改，无论多小，都必须在性能测试之前通过全量精度验证。不允许先跑性能再补精度测试——精度回归在性能数据中不可见，事后发现代价极大。

**验证步骤**（两步都必须通过才能继续）:

1. **JIT reference suite** — 覆盖所有 SUPPORTED_NUM_HEADS，快速迭代:
```bash
docker exec dyx-xllm-cann85-main bash -c "cd /export/home/dengyingxu1/projects/xllm/xllm && python3 -c \"
import sys
sys.path.insert(0, '/export/home/dengyingxu1/projects/xllm/third_party/tilelang-ascend')
from compiler.tilelang.targets.ascend.kernels.fused_gdn_gating import _run_ref_suite, SUPPORTED_NUM_HEADS
_run_ref_suite(num_batches=16, compile_max_batch=262144, softplus_beta=1.0, softplus_threshold=20.0, ref_num_heads_list=list(SUPPORTED_NUM_HEADS))
\""
```

2. **C++ AOT test** — 覆盖边界 case (B=1, B=17, B=257, B=262144, 非默认 beta/threshold 等):
```bash
docker exec dyx-xllm-cann85-main bash -c "cd /export/home/dengyingxu1/projects/xllm && python setup.py test --test-name fused_gdn_gating_wrapper_test"
```

**历史教训**: Phase 3 bulk DMA 优化引入了余数 chunk 越界 bug，导致 3/12 C++ AOT test case 失败（g 值偏差 4.73、beta 输出 nan）。由于当时只跑了 JIT benchmark 未跑精度验证，bug 直到 Phase 3.6 C++ AOT 测试阶段才被发现。如果每次改动后立即跑精度验证，可以在引入 bug 的当次就定位和修复。

## 基线数据 (num_heads=32, JIT benchmark, 50 iters)

| batch_size | num_heads | avg_us | rows/s |
|-----------|-----------|--------|--------|
| 1 | 32 | 137.29 | 7,284 |
| 16 | 32 | 137.53 | 116,338 |
| 48 | 32 | 138.23 | 347,247 |
| 256 | 32 | 138.72 | 1,845,504 |
| 1,024 | 32 | 138.49 | 7,393,869 |
| 4,096 | 32 | 138.91 | 29,486,395 |
| 8,192 | 32 | 141.17 | 58,031,242 |
| 65,536 | 32 | 1,007.67 | 65,037,014 |
| 262,144 | 32 | 4,002.20 | 65,499,951 |

## 优化后数据 (Phase 1+2, num_heads=32, JIT benchmark, 50 iters)

| batch_size | num_heads | avg_us | rows/s | speedup |
|-----------|-----------|--------|--------|---------|
| 1 | 32 | 138.95 | 7,197 | 0.99x |
| 16 | 32 | 139.09 | 115,031 | 0.99x |
| 48 | 32 | 138.67 | 346,136 | 1.00x |
| 256 | 32 | 139.36 | 1,837,010 | 1.00x |
| 1,024 | 32 | 139.13 | 7,359,805 | 1.00x |
| 4,096 | 32 | 139.41 | 29,381,485 | 1.00x |
| 8,192 | 32 | 139.72 | 58,632,589 | 1.01x |
| 65,536 | 32 | 366.00 | 179,058,270 | **2.75x** |
| 262,144 | 32 | 1,405.92 | 186,457,457 | **2.85x** |

## Phase 3 数据: Fused vs Naive PyTorch (JIT benchmark, 50 iters)

### H=32

| batch_size | fused_us | naive_us | fused/naive |
|-----------|----------|----------|-------------|
| 16 | 139 | 68 | 0.49x |
| 48 | 138 | 67 | 0.50x |
| 1,024 | 138 | 68 | 0.49x |
| 4,096 | 139 | 72 | 0.52x |
| 16,384 | 137 | 74 | 0.54x |
| 65,536 | 139 | 93 | 0.67x |
| 262,144 | **258** | 236 | **0.92x** |

### H=128

| batch_size | fused_us | naive_us | fused/naive |
|-----------|----------|----------|-------------|
| 16 | 138 | 67 | 0.48x |
| 48 | 138 | 64 | 0.46x |
| 1,024 | 138 | 63 | 0.45x |
| 4,096 | 138 | 70 | 0.50x |
| 16,384 | 140 | 93 | 0.67x |
| 65,536 | **193** | 236 | **1.22x** |
| 262,144 | **844** | 1,257 | **1.49x** |

### 结论

- H=128 大 batch: 融合 kernel 已优于 naive (1.49x at 262144)
- H=32 大 batch: 接近持平 (0.92x at 262144)
- 小 batch JIT benchmark 慢: ~137us 固定开销来自 Cython wrapper Python 对象操作，非 kernel dispatch 本身
- 绕过 Cython wrapper 直接调用: fused kernel **14.2us** vs naive **66.7us** → **4.7x faster** (B=4096 H=32)
- C++ AOT 生产路径无 Cython 开销，预期全 batch 范围优于 naive
- Phase 3 vs Phase 2 (H=32, B=262144): 1406us → 258us (**5.4x** 提升)

## msprof 实测数据 (Phase 3, H=32)

### B=4096, H=32 (60次调用, 59 warm)

| 指标 | 平均 | min | max | 说明 |
|------|------|-----|-----|------|
| Task Duration (设备执行) | 11.6us | 11.3 | 11.9 | NPU 实际执行时间 |
| Task Wait Time (host dispatch) | 142.4us | 105.1 | 235.8 | host dispatch→device 开始 |
| E2E per invocation | 154.0us | — | — | Duration + WaitTime |
| aiv_vec | 2.9us | 2.9 | 2.9 | VEC 计算 (25%) |
| aiv_mte2 | 2.2us | 2.0 | 2.4 | GM→UB DMA (19%) |
| aiv_mte3 | 0.5us | 0.5 | 0.6 | UB→GM DMA (4%) |
| aiv_scalar | 2.9us | 2.8 | 3.0 | 循环/索引 (25%) |

### B=16384, H=32 (60次调用, 59 warm)

| 指标 | 平均 | min | max | 说明 |
|------|------|-----|-----|------|
| Task Duration (设备执行) | 24.0us | 23.7 | 24.4 | NPU 实际执行时间 |
| Task Wait Time (host dispatch) | 123.2us | 99.1 | 229.1 | host dispatch→device 开始 |
| E2E per invocation | 147.2us | — | — | Duration + WaitTime |
| aiv_vec | 8.1us | 8.1 | 8.1 | VEC 计算 (34%) |
| aiv_mte2 | 7.2us | 7.1 | 7.3 | GM→UB DMA (30%) |
| aiv_mte3 | 2.3us | 2.2 | 2.4 | UB→GM DMA (10%) |
| aiv_scalar | 4.0us | 3.8 | 4.1 | 循环/索引 (17%) |

### JIT Benchmark 全景 (H=32, 100 iters)

| batch_size | fused_us | naive_us | speedup | 分析 |
|-----------|----------|----------|---------|------|
| 16 | 137.6 | 63.2 | 0.46x | dispatch 主导, device ~2us |
| 48 | 136.9 | 62.6 | 0.46x | dispatch 主导 |
| 256 | 136.9 | 63.3 | 0.46x | dispatch 主导 |
| 1,024 | 139.5 | 63.2 | 0.45x | dispatch 主导 |
| 4,096 | 137.6 | 60.7 | 0.44x | dispatch 142us, device 11.6us |
| 16,384 | 138.2 | 68.8 | 0.50x | dispatch 123us, device 24us |
| 65,536 | 139.2 | 89.7 | 0.64x | device 开始有感 |
| 262,144 | 256.4 | 233.7 | 0.91x | device ~120us, dispatch 占比降到 ~53% |

### msprof 分析结论

**重要发现: JIT benchmark 的 ~137us 固定开销来自 Python Cython wrapper，非 kernel dispatch 本身。**

| 调用路径 | B=4096 H=32 E2E | Host-only | 说明 |
|----------|----------------|-----------|------|
| JIT (Cython wrapper) | 136.4us | 136.3us | `CythonKernelWrapper.forward()` 每次调用做 tensor list 构建、dtype/device 检查、ctypes 转换等 Python 对象操作 |
| JIT (直接 ctypes lib.call) | **14.2us** | 4.4us | 绕过 Cython wrapper，直接调用编译好的 .so 中 `call()` |
| **C++ AOT (生产路径)** | **20.2us** | — | `entry->fn(...)` 直接调用预编译 .so，无 Python 开销 |
| Naive PyTorch 7 ops | 51.2us (AOT bench) | — | 7 个 CANN 内置 op |

1. **Cython wrapper 占 ~122us**: JIT benchmark 中 89% 的时间花在 Python 端对象操作上
2. **绕过 Cython 后 fused kernel 比 naive 快 4.7x**: direct ctypes 14.2us vs naive 66.7us (B=4096)
3. **C++ AOT 路径确认无 Cython 开销**: B=4096 H=32 仅 20.2us，比 naive 51.2us 快 **2.53x**
4. **device 执行很快**: B=4096 仅 9.8us, B=16384 仅 24us
5. **msprof Task Wait Time (~120-142us) 包含 Cython wrapper 开销**: 因为 msprof 采集在 JIT 路径下进行

## C++ AOT Benchmark (生产路径, 100 iters)

### H=32

| batch_size | fused_us | naive_us | speedup |
|-----------|----------|----------|---------|
| 16 | 17.5 | 49.1 | **2.80x** |
| 48 | 17.7 | 50.5 | **2.84x** |
| 1,024 | 18.1 | 50.5 | **2.80x** |
| 4,096 | 20.2 | 51.2 | **2.53x** |
| 16,384 | 29.8 | 71.0 | **2.38x** |
| 65,536 | 77.1 | 91.8 | **1.19x** |
| 262,144 | 267.5 | 239.1 | 0.89x |

### H=128

| batch_size | fused_us | naive_us | speedup |
|-----------|----------|----------|---------|
| 4,096 | 23.7 | 68.6 | **2.90x** |
| 65,536 | 196.9 | 241.0 | **1.22x** |
| 262,144 | 872.9 | 1,315.2 | **1.51x** |

### 结论

- **小~中 batch (B≤16384)**: fused kernel 比 naive **2.4-2.9x faster**，C++ AOT dispatch 仅 ~17-30us
- **大 batch H=128 (B=65536-262144)**: fused 仍优于 naive (1.22-1.51x)
- **大 batch H=32 (B=262144)**: fused 略慢 (0.89x)，后续 Phase 4 双缓冲实测也未带来改善
- **典型生产 shape (B=4096, H=32)**: fused **20.2us** vs naive **51.2us** = **2.53x faster**

## Phase 3.6: Bulk DMA 余数 chunk 修复

**问题**: bulk DMA 路径在最后一个 chunk 不满 rows_per_iter 行时，仍执行 multi-row DMA，导致越界读写。

**影响**: B 不是 rows_per_iter 倍数时，输出 g 错误或 beta 为 nan。C++ AOT test 中 3/12 case 失败：
- `tiny_b1_h16` (B=1, R=32): g mismatch, max_diff=4.73
- `medium_b257_h128` (B=257, R=8): beta nan
- `custom_beta2_threshold0p5_b33_h64` (B=33, R=32): beta nan

**修复**: 在 bulk DMA 循环内，使用 `T.If`/`T.Then`/`T.Else` 判断当前 chunk 是否完整：
- 完整 chunk: 走 multi-row bulk DMA (快速路径)
- 余数 chunk: 逐行 per-row DMA (与 non-bulk 路径一致)

修复后 C++ AOT 全 12 test case 通过，JIT reference suite 全 10 种 num_heads 通过。

### 待解决

1. Phase 3 改动未 commit 到 git
2. B=262144 H=32 仍略慢 naive (0.89x)，Phase 4 双缓冲已验证无收益，需要寻找新的优化方向

## Phase 3.7: 小 batch task 利用率优化

**问题**: msprof 设备级对比发现 H=32 小 batch (B=1~32) TileLang 比 Triton 慢。

修复前 msprof 数据 (H=32):
- B=16: TileLang 7.46us vs Triton 4.28us (Triton/TL = 0.57, TileLang 慢!)
- B=48: TileLang 7.78us vs Triton 8.69us (Triton/TL = 1.12, TileLang 胜)

**根因**: task 利用率崩塌。H=32 bulk DMA 路径 `rows_per_iter=32`，当 B<32 时只有 1 个 chunk，只有 1 个 task 在工作，其余 task 全部空闲。

| B | block_num | task_num | total_chunks | 工作 task 数 | 利用率 |
|---|-----------|----------|--------------|-------------|--------|
| 1 | 1 | 2 | 1 | 1 | 50% |
| 2 | 2 | 4 | 1 | 1 | 25% |
| 4 | 4 | 8 | 1 | 1 | 12.5% |
| 16 | 16 | 32 | 1 | 1 | 3.1% |
| 32 | 32 | 64 | 1 | 1 | 1.6% |

Triton 在 B≤48 时每个 program 只处理 1 行 (BLK_BATCHES=1)，B=16 就有 16 个 program 并行。

**修复**: 在 `build_fused_gdn_gating_kernel()` 中，当 `batch_size <= rows_per_iter` 时降低 `rows_per_iter`：

```python
if batch_size <= rows_per_iter:
    rows_per_iter = max(1, batch_size // block_num)
    use_bulk_dma = _can_use_bulk_dma(num_heads, rows_per_iter)
    if use_bulk_dma:
        ub_tensor_dim = num_heads
    else:
        ub_tensor_dim = _align_count_to_vector_bytes(num_heads, acc_dtype)
```

条件 `batch_size <= rows_per_iter` 确保只影响小 batch，B>rows_per_iter 的场景（如 B=48 H=32）不受影响。

**精度验证**: 修复后 `_run_ref_suite` 全 10 种 SUPPORTED_NUM_HEADS 通过 (B=1,2,4,8,16,32)。

### msprof 设备级对比: TileLang vs Triton (Phase 3.7, 修复后)

#### H=32

| batch | TileLang avg_us | Triton avg_us | Triton/TileLang | 分析 |
|-------|----------------|---------------|-----------------|------|
| 1 | 1.84 | 2.43 | 1.33 | TileLang 胜 |
| 2 | 1.89 | 2.50 | 1.32 | TileLang 胜 |
| 4 | 2.18 | 2.66 | 1.22 | TileLang 胜 |
| 8 | 2.43 | 3.13 | 1.29 | TileLang 胜 |
| 16 | 3.24 | 4.06 | 1.25 | TileLang 胜 (修复前 0.57!) |
| 32 | 5.04 | 7.09 | 1.41 | TileLang 胜 |
| 48 | 7.78 | 8.91 | 1.14 | TileLang 胜 (无退化) |
| 1,024 | 9.56 | 12.74 | 1.33 | TileLang 胜 |
| 4,096 | 12.63 | 21.55 | 1.71 | TileLang 胜 |
| 16,384 | 24.88 | 55.53 | 2.23 | TileLang 胜 |
| 65,536 | 72.74 | 228.50 | 3.14 | TileLang 胜 |
| 262,144 | 269.20 | 791.25 | 2.94 | TileLang 胜 |

#### H=128

| batch | TileLang avg_us | Triton avg_us | Triton/TileLang | 分析 |
|-------|----------------|---------------|-----------------|------|
| 1 | 2.07 | 6.84 | 3.31 | TileLang 胜 |
| 2 | 2.12 | 6.92 | 3.26 | TileLang 胜 |
| 4 | 2.37 | 7.06 | 2.98 | TileLang 胜 |
| 8 | 2.62 | 7.53 | 2.87 | TileLang 胜 |
| 16 | 3.46 | 12.56 | 3.63 | TileLang 胜 |
| 32 | 5.13 | 20.52 | 4.00 | TileLang 胜 |
| 48 | 6.84 | 30.60 | 4.47 | TileLang 胜 |
| 1,024 | 9.25 | 38.13 | 4.12 | TileLang 胜 |
| 4,096 | 18.89 | 64.26 | 3.40 | TileLang 胜 |
| 16,384 | 55.23 | 182.31 | 3.30 | TileLang 胜 |
| 65,536 | 198.11 | 653.80 | 3.30 | TileLang 胜 |
| 262,144 | 880.97 | 2,167.44 | 2.46 | TileLang 胜 |

### Phase 3.7 结论

- **修复前**: H=32 小 batch (B=1~32) TileLang 比 Triton 慢 (Triton/TL < 1.0)
- **修复后**: **TileLang 在全 batch 范围 (B=1~262144) 全面优于 Triton**
- H=32: Triton/TileLang = 1.14~3.14x (TileLang 快 14%~214%)
- H=128: Triton/TileLang = 2.46~4.47x (TileLang 快 146%~347%)
- B=48 H=32 无退化: 7.78us (与修复前一致)
- B≥48 的大 batch 场景不受影响

## Benchmark 基础设施

### 文件

| 文件 | 作用 |
|------|------|
| `benchmarks/fused_gdn_gating_python_perf_compare.py` | TileLang vs Triton Python-side timing + 精度比较 |
| `benchmarks/kernel_msprof_task_duration.py` | msprof 编排: 运行 worker → 解析 trace JSON → 提取设备级 task duration |
| `run_fused_gdn_gating.sh` | 4 步集成脚本: ref check → codegen stats → python perf → msprof benchmark |

### 测试范围

- **batch sizes**: 1, 2, 4, 8, 16, 32, 48, 1024, 4096, 16384, 65536, 262144
- **num_heads**: 32, 128
- **compile_max_batch**: 262144
- **warmup_iters**: 20 (python perf) / 20 (msprof)
- **measure_iters**: 200 (python perf) / 100 (msprof)

### msprof 对比方法

`kernel_msprof_task_duration.py` 通过以下步骤获取设备级 kernel 执行时间:

1. 对每个 (num_heads, num_batches) 组合，启动 worker 在 msprof 下运行
2. Worker 执行 warmup + measure 循环，同时运行 TileLang 和 Triton kernel
3. 解析 msprof 输出的 `msprof_*.json` trace 文件
4. 按 kernel 名称匹配 trace event (`fused_gdn_gating_kernel_kernel` vs `vllm_fused_gdn_gating_kernel`)
5. 取最后 measure_iters 个 event 的 `dur` 字段统计 avg/min/max
6. 计算 Triton/TileLang 比值

## Phase 4: 双缓冲实测结果

### 实验目标

验证 `T.Pipelined(num_stages=2)` 的双缓冲是否能改善 bulk DMA 路径下的设备执行时间，尤其是 H=32 的大 batch 场景。

### 验证步骤

Phase 4 按完整流程执行，4 个步骤全部通过:

1. 精度验证
2. codegen 检查
3. Python perf compare
4. msprof device-level 对比

### Codegen 结果

双缓冲 flag 模式已经生成:

| 指标 | 结果 | 说明 |
|------|------|------|
| `SetFlag` | 7 | 存在多阶段 producer/consumer 同步 |
| `WaitFlag` | 7 | 与 `SetFlag` 数量匹配 |

这说明 Phase 4 的代码路径确实进入了双缓冲同步逻辑，不是“代码没生效”。

### msprof device-level 对比（Phase 4 vs Phase 3.7）

#### H=32

| batch | Phase 4 (us) | Phase 3.7 (us) | 变化 | Triton/TileLang |
|------|--------------|----------------|------|-----------------|
| 1 | 1.85 | 1.84 | 基本持平 | 1.32x |
| 16 | 3.62 | 3.24 | +12% | 1.18x |
| 48 | 8.74 | 7.78 | +12% | 1.07x |
| 1,024 | 9.89 | 9.56 | +3% | 1.33x |
| 4,096 | 12.97 | 12.63 | +3% | 1.69x |
| 16,384 | 25.50 | 24.88 | +2% | 2.25x |
| 65,536 | 74.78 | 72.74 | +3% | 3.03x |
| 262,144 | 274.54 | 269.20 | +2% | 3.01x |

#### H=128

H=128 的结果与 Phase 3.7 也大体相当，没有观察到显著提升。

### 结论

- **双缓冲没有带来预期提速**: H=32 全范围均未快于 Phase 3.7，慢 2-12%
- **最差点出现在中小 batch**: B=16 和 B=48 均退化 12%
- **大 batch 也没有收益**: B=262144 仍慢 2%
- **H=128 同样没有明显改善**

当前判断是:

1. DMA load 时间本身已经被现有流水较好隐藏，额外重叠空间有限
2. `SetFlag/WaitFlag` 的同步成本高于潜在的 DMA/计算重叠收益
3. Phase 3.7 已经是更优基线，Phase 4 双缓冲应视为一次已完成但否定的尝试

### 对后续工作的建议

- 不要再把 Phase 4 双缓冲当作 H=32 的默认优化方向
- 后续继续优化时，应以 **Phase 3.7** 作为性能基线
- 更值得继续排查的方向:
  - H=32 大 batch 下是否仍存在额外 scalar/调度开销
  - bulk DMA chunk 粒度是否还有更合适的拆分方式
  - 是否能减少非必要同步，而不是增加新的 flag 协调

### 一句话结论

**Phase 4 双缓冲 codegen 正常，但 msprof 实测无收益，H=32 反而慢 2-12%；原因更可能是 flag 同步开销抵消了 DMA/计算重叠收益，因此当前最佳版本仍是 Phase 3.7。**
