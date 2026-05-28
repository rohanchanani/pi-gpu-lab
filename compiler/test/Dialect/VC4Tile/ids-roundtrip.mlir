// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @ids
vc4tile.kernel @ids attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  // CHECK: vc4tile.program_id
  %pid = vc4tile.program_id : i32
  // CHECK: vc4tile.block_id
  %bid = vc4tile.block_id : i32
  // CHECK: vc4tile.warp_id
  %wid = vc4tile.warp_id : i32
  // CHECK: vc4tile.lane_id
  %lid = vc4tile.lane_id : i32
  // CHECK: vc4tile.lane_range
  %lanes = vc4tile.lane_range : vector<16xi32>
  // CHECK: vc4tile.thread_id
  %tid = vc4tile.thread_id : vector<16xi32>
  // CHECK: vc4tile.mask_all
  %all = vc4tile.mask_all
  // CHECK: vc4tile.tail_mask
  %tail = vc4tile.tail_mask %pid, %lid : i32, i32
  vc4tile.return
}
