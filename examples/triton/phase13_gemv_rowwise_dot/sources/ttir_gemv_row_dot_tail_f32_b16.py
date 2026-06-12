import triton
import triton.language as tl


@triton.jit
def ttir_gemv_row_dot_tail_f32_b16_kernel(
    A, X, Y, K, ROWS, LDA, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    k_mask = offs < K
    a = tl.load(A + row * LDA + offs, mask=k_mask, other=0.0)
    x = tl.load(X + offs, mask=k_mask, other=0.0)
    acc = tl.sum(a * x, axis=0)
    if row < ROWS:
        tl.store(Y + row, acc)
