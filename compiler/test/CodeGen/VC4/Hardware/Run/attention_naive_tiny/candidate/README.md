
attention_naive_tiny candidate

A compiler-generated candidate for this test should implement the tiny full-attention baseline.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns query rows round-robin by QPU id. Lanes correspond to value dimension d for the output vector. For d < 16, inactive lanes must not store beyond the row width.

For each query row, compute up to 7 key scores, apply numerically stable softmax, and accumulate the output value vector from V. The first implementation may use in-register or VPM temporaries for the small score/probability list. VPM/VDW setup, access, and stores should remain mutex-protected.

The public launch API exposes only semantic tensors and dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler registers.
