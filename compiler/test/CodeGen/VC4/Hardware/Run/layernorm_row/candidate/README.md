
layernorm_row candidate

A compiler-generated candidate for this test should implement small row-local layer normalization.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns rows round-robin by QPU id. For each row, one QPU warp loads up to 16 active values, masks inactive lanes, reduces the sum to a scalar mean, reduces squared deviations to a scalar variance, uses SFU reciprocal-square-root for var + epsilon, and stores normalized active elements with dynamic VDW depth equal to width.

The public launch API exposes only semantic tensors, row/width dimensions, epsilon, gamma, and beta. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, SFU details, or scheduler registers.
