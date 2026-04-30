
gemv_naive_tail

Validates naive matrix-vector multiplication with arbitrary row and column tails.

For each launch, the semantic operation is:

y[row] = sum_{col=0..N-1} A[row * N + col] * x[col]

Inputs and outputs are 32-bit floats. The test includes zero-row, zero-column, small-tail, lane-width, and multi-vector-column cases. If N=0, every output row is zero. The output guard region past M must remain unchanged.

The reference bundle exposes only semantic buffers and dimensions through:

gemv_naive_tail_prepare

gemv_naive_tail_launch

gemv_naive_tail_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=gemv_naive_tail

status=PASS

cases=9

total_mismatches=0

sentinel_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

max_m=25

max_n=33

runtime_allocations=1

runtime_launches=9

max_abs_diff <= 0.001
