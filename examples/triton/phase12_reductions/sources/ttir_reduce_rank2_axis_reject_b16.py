import triton
import triton.language as tl


@triton.jit
def ttir_reduce_rank2_axis_reject_b16_kernel(
    x_ptr, out_ptr, ncols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * ncols + pid * BLOCK_N + cols
    mask = cols < ncols
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    totals = tl.sum(vals, axis=1)
    tl.store(out_ptr + pid * BLOCK_M + tl.arange(0, BLOCK_M), totals)
