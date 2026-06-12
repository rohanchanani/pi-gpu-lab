import triton
import triton.language as tl


@triton.jit
def ttir_gemv_row_dot_i32_b16_kernel(A, X, Y, K, LDA, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    a = tl.load(A + row * LDA + offs, mask=mask, other=0)
    x = tl.load(X + offs, mask=mask, other=0)
    acc = tl.sum(a * x, axis=0)
    tl.store(Y + row, acc)
