import triton
import triton.language as tl


@triton.jit
def ttir_cf_while_loop_b16_kernel(
    x_ptr, out_ptr, n_elements, trip_count, BLOCK_SIZE: tl.constexpr
):
    # Phase 8.5 controlled fixture: scalar coherent while should emit scf.while.
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offsets = pid * BLOCK_SIZE + lanes
    mask = offsets < n_elements
    acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    i = 0
    while i < trip_count:
        acc = acc + 1.0
        i += 1
    tl.store(out_ptr + offsets, acc, mask=mask)
