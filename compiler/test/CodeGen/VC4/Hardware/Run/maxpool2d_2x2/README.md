
maxpool2d_2x2

Validates a canonical 2x2 max-pooling primitive over row-major float images, including odd-width and odd-height tail windows.

For input shape height x width, the logical output shape is:

out_w = ceil(width / 2)

out_h = ceil(height / 2)

Each output element is the maximum of valid in-bounds elements from the 2x2 input window starting at (2 * oy, 2 * ox). Partial windows at odd image edges use only valid input elements. Zero width or zero height produces zero logical outputs and leaves output guards unchanged.

The reference bundle exposes only semantic buffers and dimensions through:

maxpool2d_2x2_prepare

maxpool2d_2x2_launch

maxpool2d_2x2_shutdown

The stable oracle checks the final VC4_TEST_RESULT fields:

name=maxpool2d_2x2

status=PASS

cases=8

total_mismatches=0

sentinel_mismatches=0

launch_failures=0

active_qpus=12

lanes=16

max_width=31

max_height=19

runtime_allocations=1

runtime_launches=8

checked_elements=287

max_abs_diff=0.0
