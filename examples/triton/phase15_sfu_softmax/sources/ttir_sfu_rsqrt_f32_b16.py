import triton
import triton.language as tl


@triton.jit
def ttir_sfu_rsqrt_f32_b16(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    active_x = tl.where(mask, x, 1.0)
    y = tl.rsqrt(active_x)
    tl.store(out_ptr + offs, y, mask=mask)
