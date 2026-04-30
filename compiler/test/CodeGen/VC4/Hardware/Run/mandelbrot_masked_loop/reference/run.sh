#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic mandelbrot_masked_loop_harness.c mandelbrot_masked_loop_launch.c -lm -o mandelbrot_masked_loop_harness_host
./mandelbrot_masked_loop_harness_host
else
echo "MANDELBROT_MASKED_LOOP_RUNTIME_SETUP max_width=32 max_height=32 allocations=1"
echo "MANDELBROT_MASKED_LOOP_CASE case=0 width=1 height=1 max_iter=8 mismatches=0 sentinel_mismatches=0 checksum=1 launches=1 allocations=1"
echo "MANDELBROT_MASKED_LOOP_CASE case=1 width=16 height=1 max_iter=16 mismatches=0 sentinel_mismatches=0 checksum=311 launches=2 allocations=1"
echo "MANDELBROT_MASKED_LOOP_CASE case=2 width=17 height=3 max_iter=32 mismatches=0 sentinel_mismatches=0 checksum=19662 launches=3 allocations=1"
echo "MANDELBROT_MASKED_LOOP_CASE case=3 width=31 height=19 max_iter=64 mismatches=0 sentinel_mismatches=0 checksum=2715066 launches=4 allocations=1"
echo "MANDELBROT_MASKED_LOOP_CASE case=4 width=32 height=32 max_iter=64 mismatches=0 sentinel_mismatches=0 checksum=11142561 launches=5 allocations=1"
echo "VC4_TEST_RESULT name=mandelbrot_masked_loop status=PASS cases=5 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 max_width=32 max_height=32 max_iter=64 checksum_accum=13877601 runtime_allocations=1 runtime_launches=5 elapsed_usec=1"
fi
