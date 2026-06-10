import triton
import triton.language as tl


@triton.jit
def mixed_ttir_multi_axis_cf_tail_b16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    trip_count,
    flag,
    threshold,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    nprog0 = tl.num_programs(axis=0)
    nprog1 = tl.num_programs(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    block_id = (pid2 * nprog1 + pid1) * nprog0 + pid0
    offsets = block_id * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    acc = x
    i = 0
    while i < trip_count:
        acc = acc + y
        i += 1

    if flag > 0:
        branch_value = acc * scale
    else:
        branch_value = acc - y

    selected = tl.where(branch_value > threshold, branch_value, y)
    tl.store(out_ptr + offsets, selected, mask=mask)
