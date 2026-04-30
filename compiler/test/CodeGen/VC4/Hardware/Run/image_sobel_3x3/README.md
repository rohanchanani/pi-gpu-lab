
image_sobel_3x3

Validates a naive grayscale Sobel edge-detection kernel shape with 3x3 neighborhood loads, clamp-to-edge boundary handling, exact integer arithmetic, and coalesced output semantics.

The reference bundle exposes only semantic image buffers and dimensions through:

image_sobel_3x3_prepare

image_sobel_3x3_launch

image_sobel_3x3_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=image_sobel_3x3

status=PASS

cases=6

total_mismatches=0

sentinel_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

max_width=31

max_height=19

runtime_allocations=1

runtime_launches=6

The deterministic cases cover image shapes (1,1), (2,3), (7,5), (16,16), (17,9), and (31,19) with smooth, checkerboard, sparse impulse, and affine patterns.
