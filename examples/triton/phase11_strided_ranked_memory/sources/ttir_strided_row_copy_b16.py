import triton
import triton.language as tl


@triton.jit
def ttir_strided_row_copy_b16_kernel(
    a_ptr, out_ptr, ncols, lda, ldo, BLOCK_SIZE: tl.constexpr
):
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    col = pid_col * BLOCK_SIZE + lanes
    in_idx = row * lda + col
    out_idx = row * ldo + col
    mask = col < ncols
    values = tl.load(a_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, values, mask=mask)
