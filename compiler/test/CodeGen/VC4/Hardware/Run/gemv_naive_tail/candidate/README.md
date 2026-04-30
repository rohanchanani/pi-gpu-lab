
gemv_naive_tail candidate

A compiler-generated candidate for this test should implement naive row-wise matrix-vector multiplication.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns output rows round-robin by QPU id. For each row, lanes iterate over columns in chunks of 16, load A[row, col + lane] and x[col + lane], mask the final tail, accumulate lane partial sums, reduce across lanes, and store one scalar row result. VPM/VDW setup and stores remain mutex-protected.

The public launch API exposes only semantic matrix/vector buffers and dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler details.
