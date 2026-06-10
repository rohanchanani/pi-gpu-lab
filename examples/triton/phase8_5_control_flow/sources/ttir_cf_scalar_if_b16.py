import triton
import triton.language as tl


@triton.jit
def ttir_cf_scalar_if_b16_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, flag, BLOCK_SIZE: tl.constexpr
):
    # Phase 8.5 controlled fixture: scalar runtime if/else should emit scf.if.
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    if flag > 0:
        result = x + y
    else:
        result = x - y
    tl.store(out_ptr + offsets, result, mask=mask)
