import triton
import triton.language as tl


@triton.jit
def scalar_if_probe_kernel(x_ptr, out_ptr, n_elements, flag, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if flag > 0:
        y = x + 1.0
    else:
        y = x + 2.0
    tl.store(out_ptr + offsets, y, mask=mask)
