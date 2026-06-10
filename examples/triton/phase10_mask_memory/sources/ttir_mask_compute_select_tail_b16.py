import triton
import triton.language as tl


@triton.jit
def ttir_mask_compute_select_tail_b16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    selected = tl.where(x < threshold, x, y)
    tl.store(out_ptr + offsets, selected, mask=mask)
