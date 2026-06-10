import triton
import triton.language as tl


@triton.jit
def ttir_multi_axis_pid2d_b16_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    nprog0 = tl.num_programs(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    block_id = pid1 * nprog0 + pid0
    offsets = block_id * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    encoded = pid1 * 100000 + pid0 * 1000 + lanes
    tl.store(out_ptr + offsets, encoded, mask=mask)
