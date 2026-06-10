import triton
import triton.language as tl


@triton.jit
def mixed_ttir_reduction_axes_mask_cf_strided_b16_kernel(
    x_ptr, out_ptr, ncols, nrows, lda, BLOCK_SIZE: tl.constexpr
):
    block_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    num_col_blocks = tl.num_programs(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    cols = block_col * BLOCK_SIZE + lanes
    idx = row * lda + cols
    mask = cols < ncols
    vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    if row < nrows:
        tl.store(out_ptr + row * num_col_blocks + block_col, total)
