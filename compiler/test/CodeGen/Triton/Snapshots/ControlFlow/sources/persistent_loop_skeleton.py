import triton
import triton.language as tl


@triton.jit
def persistent_loop_skeleton_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    num_tiles,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, loop_unroll_factor=1, flatten=True):
        offsets = tile_id * BLOCK_SIZE + lanes
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x + tile_id.to(tl.float32)
        tl.store(out_ptr + offsets, y, mask=mask)
