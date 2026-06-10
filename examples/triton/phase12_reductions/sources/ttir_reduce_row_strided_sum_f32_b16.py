import triton
import triton.language as tl


@triton.jit
def ttir_reduce_row_strided_sum_f32_b16_kernel(
    x_ptr, out_ptr, ncols, lda, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    idx = row * lda + lanes
    mask = lanes < ncols
    vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr + row, total)
