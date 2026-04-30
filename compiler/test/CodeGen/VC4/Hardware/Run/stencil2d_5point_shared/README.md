
stencil2d_5point_shared

This hardware-run litmus validates a fixed shared-memory-style five-point stencil tile shaped for the VC4 QPU execution model.

The intended semantic tile is a 12 x 16 staged VPM region containing a one-cell halo. The interior 10 x 14 region is checked against a CPU reference with clamp-to-edge boundary behavior. The public launch API exposes only the semantic input image, output image, image dimensions, tile origin, and stencil weights; VPM rows, semaphore IDs, uniform packing, and launch details remain private to the reference launcher.

The stable oracle checks the final VC4_TEST_RESULT line for:

name=stencil2d_5point_shared

status=PASS

cases=3

total_mismatches=0

sentinel_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

warps_per_block=12

runtime_allocations=1

runtime_launches=3

timeouts=0

errstat_relevant_changed=0

max_abs_diff <= 0.001
