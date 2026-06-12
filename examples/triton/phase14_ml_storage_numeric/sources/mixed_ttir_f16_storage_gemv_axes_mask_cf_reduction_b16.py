import triton
import triton.language as tl


@triton.jit
def mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16(A, X, Y, partials, rows: tl.constexpr, K: tl.constexpr, LDA: tl.constexpr, NUM_KBLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    kblock = tl.program_id(0)
    row = tl.program_id(1)
    offs = kblock * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    a = tl.load(A + row * LDA + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(a * x, axis=0)
    if kblock == 0:
        tl.store(Y + row, acc, mask=row < rows)
    tl.store(partials + row * NUM_KBLOCKS + kblock, acc, mask=row < rows)
