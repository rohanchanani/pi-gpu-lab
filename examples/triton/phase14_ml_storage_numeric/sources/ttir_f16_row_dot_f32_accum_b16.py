import triton
import triton.language as tl


@triton.jit
def ttir_f16_row_dot_f32_accum_b16(A, X, Y, rows: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    a = tl.load(A + row * LDA + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(a * x, axis=0)
    tl.store(Y + row, acc, mask=row < rows)
