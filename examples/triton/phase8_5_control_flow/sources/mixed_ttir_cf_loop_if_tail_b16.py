import triton
import triton.language as tl


@triton.jit
def mixed_ttir_cf_loop_if_tail_b16_kernel(
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
    # Phase 8.5 mixed fixture: tail-masked loads/stores plus scalar while and
    # scalar runtime if. Body ops stay in f32 add/sub/mul/cmp/select.
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
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
