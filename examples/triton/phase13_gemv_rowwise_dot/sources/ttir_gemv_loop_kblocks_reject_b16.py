import triton
import triton.language as tl


@triton.jit
def ttir_gemv_loop_kblocks_reject_b16_kernel(
    A, X, Y, K, LDA, NUM_KBLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for kb in tl.range(0, NUM_KBLOCKS):
        offs = kb * BLOCK_SIZE + lanes
        mask = offs < K
        a = tl.load(A + row * LDA + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        acc += tl.sum(a * x, axis=0)
    tl.store(Y + row, acc)
