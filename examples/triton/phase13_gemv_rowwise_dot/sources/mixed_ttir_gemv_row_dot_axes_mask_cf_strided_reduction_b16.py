import triton
import triton.language as tl


@triton.jit
def mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_kernel(
    A, X, Y, K, LDA, ROWS, flag, bias, BLOCK_SIZE: tl.constexpr
):
    kblock = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    offs = kblock * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    a = tl.load(A + row * LDA + offs, mask=mask, other=0.0)
    x = tl.load(X + offs, mask=mask, other=0.0)
    acc = tl.sum(a * x, axis=0)
    if flag != 0:
        acc = acc + bias
    if row < ROWS:
        tl.store(Y + row, acc)
