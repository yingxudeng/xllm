import tilelang
import tilelang.language as T

DEFAULT_ASCEND_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
}


def build_test_kernel():
    @T.prim_func
    def test_copy(
        a: T.Tensor((256, 32), "bfloat16"),
        out: T.Tensor((256, 32), "bfloat16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                buf = T.alloc_shared((16, 32), "bfloat16")
                base = cid * 16
                T.copy(a[base, 0], buf)
                T.copy(buf, out[base, 0])

    return test_copy


tilelang.disable_cache()
kernel_func = build_test_kernel()
with tilelang.tvm.transform.PassContext(
    opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
):
    kernel = tilelang.engine.lower(kernel_func)
source = kernel.kernel_source
for i, line in enumerate(source.split("\n")):
    if "copy_gm" in line or "copy_ub" in line:
        print(f"{i+1:4d}: {line.strip()}")
