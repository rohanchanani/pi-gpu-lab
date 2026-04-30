
attention_qk_naive candidate

A compiler-generated candidate for this test should implement the attention QK score baseline.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns query rows round-robin by QPU id. Within each query row, lane id maps to key index k. Inactive lanes for lane >= k_len are masked and not stored.

For each feature dimension t, the kernel loads Q[q, t] and K[k, t], accumulates Q * K per lane, multiplies by the scalar scale, and stores the score row through VPM/VDW with dynamic depth equal to k_len.

The public launch API exposes only semantic tensors and dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, or scheduler registers.
