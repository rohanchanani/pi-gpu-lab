import triton
import triton.language as tl


@triton.jit
def ttir_lane_varying_stride_gather_reject_b16_kernel(
    a_ptr, out_ptr, ncols, lda, ldo, stride, BLOCK_SIZE: tl.constexpr
):
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    lanes = tl.arange(0, BLOCK_SIZE)
    col = pid_col * BLOCK_SIZE + lanes
    in_idx = row * lda + lanes * stride
    out_idx = row * ldo + col
    mask = col < ncols
    values = tl.load(a_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, values, mask=mask)
