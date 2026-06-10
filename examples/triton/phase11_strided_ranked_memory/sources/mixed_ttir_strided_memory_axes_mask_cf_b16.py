import triton
import triton.language as tl


@triton.jit
def mixed_ttir_strided_memory_axes_mask_cf_b16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    threshold,
    bias,
    flag,
    ncols,
    ldx,
    ldy,
    ldo,
    BLOCK_SIZE: tl.constexpr,
):
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    col = pid_col * BLOCK_SIZE + lanes
    x_idx = row * ldx + col
    y_idx = row * ldy + col
    out_idx = row * ldo + col
    mask = col < ncols
    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_idx, mask=mask, other=0.0)
    selected = tl.where(x < threshold, x, y)
    if flag != 0:
        selected = selected + bias
    tl.store(out_ptr + out_idx, selected, mask=mask)
