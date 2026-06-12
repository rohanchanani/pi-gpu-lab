import triton
import triton.language as tl


@triton.jit
def ttir_f32_to_i32_cast_reject_b16(x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x.to(tl.int32)
    tl.store(out_ptr + offsets, y, mask=mask)
