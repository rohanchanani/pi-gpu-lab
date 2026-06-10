import triton
import triton.language as tl


@triton.jit
def ttir_strided_row_add_b16_kernel(
    a_ptr, b_ptr, c_ptr, ncols, lda, ldb, ldc, BLOCK_SIZE: tl.constexpr
):
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    col = pid_col * BLOCK_SIZE + lanes
    a_idx = row * lda + col
    b_idx = row * ldb + col
    c_idx = row * ldc + col
    mask = col < ncols
    a = tl.load(a_ptr + a_idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_idx, mask=mask, other=0.0)
    tl.store(c_ptr + c_idx, a + b, mask=mask)
