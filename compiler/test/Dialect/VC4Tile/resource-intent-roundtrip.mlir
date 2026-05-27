// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @resource_intent_metadata
// CHECK-DAG: resource_intent =
// CHECK-DAG: uses_shared_vpm = true
// CHECK-DAG: uses_barrier = true
// CHECK-DAG: vpm_rows = 8 : i32
// CHECK-DAG: vpm_bytes = 512 : i32
// CHECK-DAG: semaphores = 4 : i32
// CHECK-DAG: warps_per_block = 4 : i32
vc4tile.kernel @resource_intent_metadata attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  public_name = "resource_intent_metadata",
  warps_per_block_max = 4 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 8 : i32,
  vpm_bytes_per_block = 512 : i32,
  resource_intent = {
    uses_shared_vpm = true,
    uses_barrier = true,
    vpm_rows = 8 : i32,
    vpm_bytes = 512 : i32,
    semaphores = 4 : i32,
    warps_per_block = 4 : i32
  }
} {
  %tile = vc4tile.shared_alloc {rows = 8 : i32, elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
