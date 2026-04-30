
image_boxblur_shared

Validates a shared-memory-shaped 3x3 grayscale box blur tile using the same public API and oracle shape expected from a VPM-staged cooperative implementation.

The semantic output is:

out[y,x] = floor((sum of 9 clamped neighbor pixels + 4) / 9)

Inputs are row-major uint32_t grayscale pixels using the low 8 bits. Each launch verifies only the fixed 10x14 output tile associated with the public tile origin, plus a sentinel guard region.

The reference bundle exposes only semantic image buffers, dimensions, and tile origin through:

image_boxblur_shared_prepare

image_boxblur_shared_launch

image_boxblur_shared_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=image_boxblur_shared

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
