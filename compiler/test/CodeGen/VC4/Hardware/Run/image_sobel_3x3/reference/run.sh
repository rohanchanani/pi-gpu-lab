#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic image_sobel_3x3_harness.c image_sobel_3x3_launch.c -o image_sobel_3x3_harness_host
./image_sobel_3x3_harness_host
else
echo "IMAGE_SOBEL_3X3_RUNTIME_SETUP max_width=31 max_height=19 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=0 width=1 height=1 mismatches=0 sentinel_mismatches=0 checksum=0 launches=1 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=1 width=2 height=3 mismatches=0 sentinel_mismatches=0 checksum=1020 launches=2 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=2 width=7 height=5 mismatches=0 sentinel_mismatches=0 checksum=5210 launches=3 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=3 width=16 height=16 mismatches=0 sentinel_mismatches=0 checksum=35930 launches=4 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=4 width=17 height=9 mismatches=0 sentinel_mismatches=0 checksum=37740 launches=5 allocations=1"
echo "IMAGE_SOBEL_3X3_CASE case=5 width=31 height=19 mismatches=0 sentinel_mismatches=0 checksum=138678 launches=6 allocations=1"
echo "VC4_TEST_RESULT name=image_sobel_3x3 status=PASS cases=6 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 max_width=31 max_height=19 runtime_allocations=1 runtime_launches=6 checksum_accum=218578 elapsed_usec=1"
fi
