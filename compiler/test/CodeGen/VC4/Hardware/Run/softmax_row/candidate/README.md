
softmax_row candidate

A compiler-generated candidate for this test should implement small row-local softmax.

The intended hardware shape uses 12 QPUs and 16 lanes. A straightforward mapping assigns rows round-robin by QPU id. For each row, one QPU warp loads up to 16 active values, masks inactive lanes to negative infinity for the max reduction, reduces the maximum, subtracts it, evaluates exponentials with SFU_EXP, reduces the exponential sum, computes the reciprocal with SFU_RECIP, multiplies active lanes by the reciprocal, and stores active elements with dynamic VDW depth equal to width.

The public launch API exposes only semantic tensors and row/width dimensions. It must not expose QPU ids, raw uniforms, VPM rows, VDW setup, SFU details, or scheduler registers.
