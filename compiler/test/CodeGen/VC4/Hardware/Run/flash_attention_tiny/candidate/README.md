
flash_attention_tiny candidate

A compiler-generated candidate for this test should implement the tiny streaming-attention baseline.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns query rows round-robin by QPU id. Lanes correspond to value dimension d for the output vector. For d < 16, inactive lanes must not store beyond the row width.

For each query row, initialize online m = -infinity, denominator l = 0, and accumulator vector acc = 0. For each key row, compute a score by reducing Q[d] * K[k,d] across lanes, update the online softmax state, load V[k,d], and update the lane-wise accumulator. Store the final accumulator vector through VPM/VDW.

The public launch API exposes only semantic tensors and dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler registers.
