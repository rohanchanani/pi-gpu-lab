import triton
import triton.language as tl


@triton.jit
def ttir_strided_row_inout_tail_b16_kernel(
    a_ptr, out_ptr, ncols, lda, ldo, scale, BLOCK_SIZE: tl.constexpr
):
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    col = pid_col * BLOCK_SIZE + lanes
    in_idx = row * lda + col
    out_idx = row * ldo + col
    mask = col < ncols
    a = tl.load(a_ptr + in_idx, mask=mask, other=0.0)
    previous = tl.load(out_ptr + out_idx, mask=mask, other=0.0)
    updated = previous + a * scale
    tl.store(out_ptr + out_idx, updated, mask=mask)
