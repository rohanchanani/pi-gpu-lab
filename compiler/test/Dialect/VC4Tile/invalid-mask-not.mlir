// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s

// CHECK: input type must be vector<16xi1>
vc4tile.kernel @invalid_mask_not attributes {
  public_name = "invalid_mask_not",
  arg_attrs = [],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %bad = vc4tile.mask_not %zero : i32 -> vector<16xi1>
  vc4tile.return
}
