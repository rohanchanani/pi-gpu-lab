import triton
import triton.language as tl


@triton.jit
def saxpy_select_b16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    a,
    threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    scaled = a * x + y
    selected = tl.where(scaled < threshold, scaled, y)
    tl.store(out_ptr + offsets, selected, mask=mask)
