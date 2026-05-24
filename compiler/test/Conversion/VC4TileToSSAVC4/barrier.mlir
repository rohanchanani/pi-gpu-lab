// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @qpu_barrier_syncthreads_vc4tile
// CHECK-SAME: vc4.launch_abi
// CHECK-DAG: #vc4.builtin_kind<logical_warp_id>
// CHECK-DAG: name = "logical_warp_id"
// CHECK-DAG: uniform_index = 0 : i32
// CHECK-DAG: #vc4.builtin_kind<warps_per_block>
// CHECK-DAG: name = "warps_per_block"
// CHECK-DAG: uniform_index = 1 : i32
// CHECK-DAG: uniform_words_per_qpu = 2 : i32
// CHECK-DAG: schedule_mode = "cooperative_block"
// CHECK-DAG: uses_barrier = true
// CHECK-DAG: semaphores_per_block = 4 : i32
// CHECK-DAG: require_full_block_residency = true
// CHECK: ssavc4.barrier
// CHECK-SAME: arrive_offset = 0 : i32
// CHECK-SAME: depart_offset = 2 : i32
// CHECK-SAME: go_offset = 1 : i32
// CHECK-SAME: reset_offset = 3 : i32
// CHECK-NEXT: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @qpu_barrier_syncthreads_vc4tile attributes {
  public_name = "qpu_barrier_syncthreads_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 12 : i32,
  uses_shared_vpm = false,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
