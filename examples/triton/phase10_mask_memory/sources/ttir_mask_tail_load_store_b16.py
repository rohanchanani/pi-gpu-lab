import triton
import triton.language as tl


@triton.jit
def ttir_mask_tail_load_store_b16_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)
