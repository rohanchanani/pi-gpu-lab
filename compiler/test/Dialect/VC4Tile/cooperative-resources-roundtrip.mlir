// RUN: vc4-opt %s | vc4-opt | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @cooperative_block_resources_roundtrip(%{{.*}}: i32)
// CHECK: schedule_mode = #vc4tile.schedule_mode<cooperative_block>
// CHECK: warps_per_block_max = 2 : i32
// CHECK: vc4tile.block_id
// CHECK: vc4tile.warp_id
// CHECK: vc4tile.thread_id
// CHECK: vc4tile.return
vc4tile.kernel @cooperative_block_resources_roundtrip attributes {
  public_name = "cooperative_block_resources_roundtrip",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32,
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %block = vc4tile.block_id : i32
  %warp = vc4tile.warp_id : i32
  %thread = vc4tile.thread_id : vector<16xi32>
  vc4tile.return
}
