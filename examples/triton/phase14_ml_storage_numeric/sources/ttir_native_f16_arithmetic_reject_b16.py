import triton
import triton.language as tl


@triton.jit
def ttir_native_f16_arithmetic_reject_b16(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)
