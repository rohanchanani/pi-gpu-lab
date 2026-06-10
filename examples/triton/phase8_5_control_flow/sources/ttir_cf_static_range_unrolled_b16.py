import triton
import triton.language as tl


@triton.jit
def ttir_cf_static_range_unrolled_b16_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Phase 8.5 optional fixture: tl.static_range is expected to specialize or
    # unroll, so this is an import regression rather than runtime CF hardware.
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    for _ in tl.static_range(0, 3, 1):
        acc = acc + 1.0
    tl.store(out_ptr + offsets, acc, mask=mask)
