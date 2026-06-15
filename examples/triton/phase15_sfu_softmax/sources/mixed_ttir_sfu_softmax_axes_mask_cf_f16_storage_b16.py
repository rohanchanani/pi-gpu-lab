import triton
import triton.language as tl


@triton.jit
def mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16(x_ptr, scale, out_ptr, audit_ptr, ncols, stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < ncols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    active_x = tl.where(mask, x * scale, -80.0)
    if group == 0:
        m = tl.max(active_x, axis=0)
        shifted = active_x - m
        e = tl.exp(shifted)
        active_e = tl.where(mask, e, 0.0)
        denom = tl.sum(active_e, axis=0)
        out = active_e / denom
        tl.store(out_ptr + row * stride + offs, out, mask=mask)
        tl.store(audit_ptr + row, denom)
