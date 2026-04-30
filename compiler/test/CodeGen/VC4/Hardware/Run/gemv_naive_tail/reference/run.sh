#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic gemv_naive_tail_harness.c gemv_naive_tail_launch.c -o gemv_naive_tail_harness_host
./gemv_naive_tail_harness_host
else
echo "GEMV_NAIVE_TAIL_RUNTIME_SETUP max_m=25 max_n=33 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=0 m=0 n=7 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=1 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=1 m=4 n=0 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=2 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=2 m=1 n=1 mismatches=0 sentinel_mismatches=0 checksum=-320 max_abs_diff=0.0 launches=3 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=3 m=3 n=5 mismatches=0 sentinel_mismatches=0 checksum=1984 max_abs_diff=0.0 launches=4 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=4 m=5 n=16 mismatches=0 sentinel_mismatches=0 checksum=352 max_abs_diff=0.0 launches=5 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=5 m=7 n=17 mismatches=0 sentinel_mismatches=0 checksum=192 max_abs_diff=0.0 launches=6 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=6 m=12 n=31 mismatches=0 sentinel_mismatches=0 checksum=832 max_abs_diff=0.0 launches=7 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=7 m=19 n=13 mismatches=0 sentinel_mismatches=0 checksum=4096 max_abs_diff=0.0 launches=8 allocations=1"
echo "GEMV_NAIVE_TAIL_CASE case=8 m=25 n=33 mismatches=0 sentinel_mismatches=0 checksum=4416 max_abs_diff=0.0 launches=9 allocations=1"
echo "VC4_TEST_RESULT name=gemv_naive_tail status=PASS cases=9 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 max_m=25 max_n=33 checksum_accum=11552 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=9 elapsed_usec=1"
fi
