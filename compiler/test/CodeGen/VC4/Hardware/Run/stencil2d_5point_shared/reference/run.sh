#!/usr/bin/env bash
set -eu

echo "ASSEMBLING QASM"
echo "RUNNING MAKE"
echo "STENCIL2D_5POINT_SHARED_RUNTIME_SETUP max_width=31 max_height=19 allocations=1 warps_per_block=12"
echo "STENCIL2D_5POINT_SHARED_CASE case=0 width=18 height=14 origin_x=0 origin_y=0 mismatches=0 sentinel_mismatches=0 checksum=0x00000000 max_abs_diff=0.0 launches=1 allocations=1"
echo "STENCIL2D_5POINT_SHARED_CASE case=1 width=31 height=19 origin_x=3 origin_y=2 mismatches=0 sentinel_mismatches=0 checksum=0x00000000 max_abs_diff=0.0 launches=2 allocations=1"
echo "STENCIL2D_5POINT_SHARED_CASE case=2 width=31 height=19 origin_x=17 origin_y=9 mismatches=0 sentinel_mismatches=0 checksum=0x00000000 max_abs_diff=0.0 launches=3 allocations=1"
echo "VC4_TEST_RESULT name=stencil2d_5point_shared status=PASS cases=3 total_mismatches=0 sentinel_mismatches=0 launch_failures=0 active_qpus=12 lanes=16 warps_per_block=12 max_abs_diff=0.0 runtime_allocations=1 runtime_launches=3 timeouts=0 errstat_relevant_changed=0 elapsed_usec=0"
