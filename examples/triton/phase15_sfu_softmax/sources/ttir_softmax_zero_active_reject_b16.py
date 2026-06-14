import triton
import triton.language as tl


@triton.jit
def ttir_softmax_zero_active_reject_b16(x_ptr, out_ptr, ncols, stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < ncols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    active_x = tl.where(mask, x, -80.0)
    m = tl.max(active_x, axis=0)
    shifted = x - m
    e = tl.exp(shifted)
    active_e = tl.where(mask, e, 0.0)
    denom = tl.sum(active_e, axis=0)
    out = active_e / denom
    tl.store(out_ptr + row * stride + offs, out, mask=mask)
