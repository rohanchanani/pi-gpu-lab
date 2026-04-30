
image_boxblur_shared candidate

A compiler-generated candidate for this test should implement one 12-warp resident cooperative block per launch.

The intended hardware shape stages a 12x16 clamped input tile into VPM rows, synchronizes all participating warps with the locked four-semaphore barrier, computes a 10x14 interior output tile, and stores the result through the conservative VPM/VDW store path.

The reference semantic transform is exact integer 3x3 box blur:

out[y,x] = floor((sum of 9 clamped neighbor pixels + 4) / 9)

Only the tile region is checked by the harness. Guard words outside the logical image output must remain sentinel.
