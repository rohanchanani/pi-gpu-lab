import triton
import triton.language as tl


@triton.jit
def ttir_column_slice_reject_b16_kernel(
    a_ptr, out_ptr, nrows, lda, ldo, col, BLOCK_SIZE: tl.constexpr
):
    pid_row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    row = pid_row * BLOCK_SIZE + lanes
    in_idx = row * lda + col
    out_idx = row * ldo + col
    mask = row < nrows
    values = tl.load(a_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, values, mask=mask)
