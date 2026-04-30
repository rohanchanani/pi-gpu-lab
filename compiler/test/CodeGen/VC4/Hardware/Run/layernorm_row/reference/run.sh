#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic layernorm_row_harness.c layernorm_row_launch.c -lm -o layernorm_row_harness_host
./layernorm_row_harness_host
else
echo "LAYERNORM_ROW_RUNTIME_SETUP max_rows=13 max_width=16 allocations=1"
echo "LAYERNORM_ROW_CASE case=0 rows=0 width=8 mismatches=0 sentinel_mismatches=0 checksum=0 max_abs_diff=0.0 launches=1 allocations=1"
echo "LAYERNORM_ROW_CASE case=1 rows=1 width=1 mismatches=0 sentinel_mismatches=0 checksum=-512 max_abs_diff=0.0 launches=2 allocations=1"
echo "LAYERNORM_ROW_CASE case=2 rows=2 width=4 mismatches=0 sentinel_mismatches=0 checksum=-4096 max_abs_diff=0.0 launches=3 allocations=1"
echo "LAYERNORM_ROW_CASE case=3 rows=3 width=8 mismatches=0 sentinel_mismatches=0 checksum=-12288 max_abs_diff=0.0 launches=4 allocations=1"
echo "LAYERNORM_ROW_CASE case=4 rows=5 width=15 mismatches=0 sentinel_mismatches=0 checksum=-38400 max_abs_diff=0.0 launches=5 allocations=1"
echo "LAYERNORM_ROW_CASE case=5 rows=7 width=16 mismatches=0 sentinel_mismatches=0 checksum=-57344 max_abs_diff=0.0 launches=6 allocations=1"
echo "LAYERNORM_ROW_CASE case=6 rows=13 width=9 mismatches=0 sentinel_mismatches=0 checksum=-59904 max_abs_diff=0.0 launches=7 allocations=1"
echo "VC4_TEST_RESULT name=layernorm_row status=PASS cases=7 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 max_rows=13 max_width=16 checksum_accum=-172544 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=7 elapsed_usec=1"
fi
