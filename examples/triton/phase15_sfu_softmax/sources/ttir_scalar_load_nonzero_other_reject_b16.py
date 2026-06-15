import triton
import triton.language as tl


@triton.jit
def ttir_scalar_load_nonzero_other_reject_b16(scale_ptr, out_ptr, n_rows, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    row_active = row < n_rows
    scale = tl.load(scale_ptr + row, mask=row_active, other=1.0)
    lanes = tl.arange(0, BLOCK_SIZE)
    offs = row * BLOCK_SIZE + lanes
    store_mask = row_active & (lanes == 0)
    tl.store(out_ptr + offs, scale, mask=store_mask)
