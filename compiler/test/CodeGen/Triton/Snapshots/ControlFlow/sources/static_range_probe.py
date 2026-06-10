import triton
import triton.language as tl


@triton.jit
def static_range_probe_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    for i in tl.static_range(0, 3, 1):
        acc = acc + i
    tl.store(out_ptr + offsets, acc, mask=mask)
