
layernorm_row

Validates a small row-wise layer normalization kernel.

For each row of width W <= 16, the semantic operation is:

mean = sum(x) / W
var = sum((x - mean)^2) / W
out = (x - mean) * rsqrt(var + epsilon) * gamma + beta

This test uses scalar gamma and beta shared across all elements. The cases cover zero rows, width one, narrow rows, tails less than 16 lanes, full 16-lane rows, and more rows than active QPUs. The output guard region past the logical rows * width elements must remain unchanged.

The reference bundle exposes only semantic buffers, dimensions, and scalar layernorm parameters through:

layernorm_row_prepare

layernorm_row_launch

layernorm_row_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=layernorm_row

status=PASS

cases=7

total_mismatches=0

sentinel_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

max_rows=13

max_width=16

runtime_allocations=1

runtime_launches=7

max_abs_diff <= 0.02
