import triton
import triton.language as tl


@triton.jit
def ttir_mask_full_no_mask_b16_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)
