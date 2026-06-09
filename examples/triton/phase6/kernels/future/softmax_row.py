import triton
import triton.language as tl


@triton.jit
def softmax_row_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_values = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=-float("inf"))
    shifted = row_values - tl.max(row_values, axis=0)
    numerators = tl.exp(shifted)
    denominator = tl.sum(numerators, axis=0)
    result = numerators / denominator
    # Later math phases must make the approximate exp/SFU policy explicit.
    tl.store(out_ptr + row * row_stride + offsets, result, mask=mask)
