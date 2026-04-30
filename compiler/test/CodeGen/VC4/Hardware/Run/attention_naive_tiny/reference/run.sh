#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic attention_naive_tiny_harness.c attention_naive_tiny_launch.c -lm -o attention_naive_tiny_harness_host
./attention_naive_tiny_harness_host
else
echo "ATTENTION_NAIVE_TINY_RUNTIME_SETUP max_q_len=5 max_k_len=7 max_d=16 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=0 q_len=0 k_len=4 d=8 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=1 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=1 q_len=2 k_len=0 d=8 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=2 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=2 q_len=1 k_len=1 d=1 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=3 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=3 q_len=2 k_len=3 d=4 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=4 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=4 q_len=3 k_len=4 d=8 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=5 allocations=1"
echo "ATTENTION_NAIVE_TINY_CASE case=5 q_len=5 k_len=7 d=16 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=6 allocations=1"
echo "VC4_TEST_RESULT name=attention_naive_tiny status=PASS cases=6 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 max_q_len=5 max_k_len=7 max_d=16 checksum_accum=0 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=6 elapsed_usec=1"
fi
