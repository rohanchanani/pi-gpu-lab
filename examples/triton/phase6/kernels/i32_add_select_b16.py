import triton
import triton.language as tl


@triton.jit
def i32_add_select_b16_kernel(
    x_ptr,
    y_ptr,
    alt_ptr,
    out_ptr,
    bias,
    threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    alt = tl.load(alt_ptr + offsets, mask=mask, other=0)
    tmp = x + y - bias
    cond = tmp > threshold
    selected = tl.where(cond, tmp, alt)
    tl.store(out_ptr + offsets, selected, mask=mask)
