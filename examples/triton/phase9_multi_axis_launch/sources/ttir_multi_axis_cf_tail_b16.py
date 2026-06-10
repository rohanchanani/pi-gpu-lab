import triton
import triton.language as tl


@triton.jit
def ttir_multi_axis_cf_tail_b16_kernel(
    x_ptr, out_ptr, n_elements, flag, BLOCK_SIZE: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    nprog0 = tl.num_programs(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    block_id = pid1 * nprog0 + pid0
    offsets = block_id * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if flag > 0:
        result = x + 1.0
    else:
        result = x - 1.0
    tl.store(out_ptr + offsets, result, mask=mask)
