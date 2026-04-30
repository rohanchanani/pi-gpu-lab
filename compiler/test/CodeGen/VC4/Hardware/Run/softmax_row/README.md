
softmax_row

Validates a small row-wise softmax kernel.

For each row of width W <= 16, the semantic operation is:

m = max(x)
e_i = exp(x_i - m)
s = sum(e_i)
out_i = e_i / s

Rows with W=0 are no-op rows. A shape with zero rows is a no-op launch. Only active lanes are written, and the output guard region past the logical rows * width elements must remain unchanged.

The reference bundle exposes only semantic buffers and dimensions through:

softmax_row_prepare

softmax_row_launch

softmax_row_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=softmax_row

status=PASS

cases=7

total_mismatches=0

sentinel_mismatches=0

row_sum_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

max_rows=13

max_width=16

runtime_allocations=1

runtime_launches=7

max_abs_diff <= 0.03
