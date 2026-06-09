import triton
import triton.language as tl


@triton.jit
def layer_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(variance + eps)
    y = centered * inv_std * weight + bias
    tl.store(out_ptr + row * row_stride + offsets, y, mask=mask)
