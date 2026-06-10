import triton
import triton.language as tl


@triton.jit
def mixed_ttir_mask_memory_cf_axes_b16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    threshold,
    bias,
    flag,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    nprog0 = tl.num_programs(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    block_id = pid1 * nprog0 + pid0
    offsets = block_id * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    selected = tl.where(x < threshold, x, y)
    if flag != 0:
        selected = selected + bias
    tl.store(out_ptr + offsets, selected, mask=mask)
