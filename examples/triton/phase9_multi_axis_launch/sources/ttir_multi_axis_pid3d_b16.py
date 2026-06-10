import triton
import triton.language as tl


@triton.jit
def ttir_multi_axis_pid3d_b16_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    nprog0 = tl.num_programs(axis=0)
    nprog1 = tl.num_programs(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    block_id = (pid2 * nprog1 + pid1) * nprog0 + pid0
    offsets = block_id * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    encoded = pid2 * 1000000 + pid1 * 10000 + pid0 * 100 + lanes
    tl.store(out_ptr + offsets, encoded, mask=mask)
