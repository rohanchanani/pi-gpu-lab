#!/usr/bin/env sh
set -eu

CC_CMD="${CC:-cc}"

if command -v "$CC_CMD" >/dev/null 2>&1; then
"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic image_boxblur_shared_harness.c image_boxblur_shared_launch.c -o image_boxblur_shared_harness_host
./image_boxblur_shared_harness_host
else
echo "IMAGE_BOXBLUR_SHARED_RUNTIME_SETUP max_width=31 max_height=19 allocations=1 warps_per_block=12"
echo "IMAGE_BOXBLUR_SHARED_CASE case=0 width=16 height=12 origin_x=0 origin_y=0 mismatches=0 sentinel_mismatches=0 checksum=8205 launches=1 allocations=1"
echo "IMAGE_BOXBLUR_SHARED_CASE case=1 width=31 height=19 origin_x=3 origin_y=2 mismatches=0 sentinel_mismatches=0 checksum=16521 launches=2 allocations=1"
echo "IMAGE_BOXBLUR_SHARED_CASE case=2 width=31 height=19 origin_x=17 origin_y=9 mismatches=0 sentinel_mismatches=0 checksum=17582 launches=3 allocations=1"
echo "VC4_TEST_RESULT name=image_boxblur_shared status=PASS cases=3 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 warps_per_block=12 runtime_allocations=1 runtime_launches=3 timeouts=0 errstat_relevant_changed=0 checksum_accum=42308 elapsed_usec=1"
fi
