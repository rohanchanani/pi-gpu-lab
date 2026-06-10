import triton
import triton.language as tl


@triton.jit
def ttir_cf_tl_range_loop_b16_kernel(
    x_ptr, out_ptr, n_elements, trip_count, BLOCK_SIZE: tl.constexpr
):
    # Phase 8.5 controlled fixture: runtime tl.range should emit scf.for.
    # The loop IV is not converted to f32; the body only adds a f32 constant.
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    for _ in tl.range(0, trip_count, 1):
        acc = acc + 1.0
    tl.store(out_ptr + offsets, acc, mask=mask)
