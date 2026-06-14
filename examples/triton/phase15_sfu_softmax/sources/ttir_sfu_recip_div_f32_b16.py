import triton
import triton.language as tl


@triton.jit
def ttir_sfu_recip_div_f32_b16(num_ptr, den_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    num = tl.load(num_ptr + offs, mask=mask, other=0.0)
    den = tl.load(den_ptr + offs, mask=mask, other=0.0)
    den_active = tl.where(mask, den, 1.0)
    y = num / den_active
    tl.store(out_ptr + offs, y, mask=mask)
