import triton
import triton.language as tl


@triton.jit
def ttir_tl_dot_still_reject_b16(A, X, Y, K, LDA, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    a_ptrs = A + row[:, None] * LDA + kk[None, :]
    x_ptrs = X + kk[:, None] + col[None, :]
    a = tl.load(a_ptrs, mask=kk[None, :] < K, other=0.0)
    x = tl.load(x_ptrs, mask=kk[:, None] < K, other=0.0)
    acc = tl.dot(a, x)
    tl.store(Y + row[:, None] * BLOCK_N + col[None, :], acc)
