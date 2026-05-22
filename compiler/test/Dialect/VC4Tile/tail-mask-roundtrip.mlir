// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tail_mask_roundtrip
vc4tile.kernel @tail_mask_roundtrip attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  public_name = "tail_mask_roundtrip",
  warps_per_block_max = 1 : i32,
  semaphores_per_block = 0 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %base = vc4tile.program_id : i32
  %limit = arith.constant 13 : i32
  // CHECK: vc4tile.tail_mask
  %tail = vc4tile.tail_mask %base, %limit : i32, i32 -> vector<16xi1>
  // CHECK: vc4tile.mask_all
  %all = vc4tile.mask_all : vector<16xi1>
  vc4tile.return
}
