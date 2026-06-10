import triton
import triton.language as tl


@triton.jit
def tl_range_loop_skeleton_kernel(x_ptr, out_ptr, n_elements, trip_count, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    acc = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    for i in tl.range(0, trip_count, 1, loop_unroll_factor=1):
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += x + i.to(tl.float32)
    tl.store(out_ptr + offsets, acc, mask=mask)
