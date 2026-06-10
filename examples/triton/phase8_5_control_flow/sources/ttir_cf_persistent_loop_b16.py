import triton
import triton.language as tl


@triton.jit
def ttir_cf_persistent_loop_b16_kernel(
    x_ptr, out_ptr, n_elements, total_blocks, BLOCK_SIZE: tl.constexpr
):
    # Phase 8.5 controlled fixture: persistent pid-strided loop skeleton.
    # This proves the control skeleton only; the body is a masked f32 add.
    block = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    while block < total_blocks:
        offsets = block * BLOCK_SIZE + lanes
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = x + 1.0
        tl.store(out_ptr + offsets, result, mask=mask)
        block += stride
