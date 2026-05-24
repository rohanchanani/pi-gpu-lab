// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @qpu_barrier_syncthreads_vc4tile
// SSAVC4: #vc4.builtin_kind<logical_warp_id>
// SSAVC4: #vc4.builtin_kind<warps_per_block>
// SSAVC4-DAG: uses_barrier = true
// SSAVC4-DAG: semaphores_per_block = 4 : i32
// SSAVC4: ssavc4.barrier
// SSAVC4-SAME: arrive_offset = 0 : i32
// SSAVC4: ssavc4.thread_end

// VC4-LABEL: vc4.func @qpu_barrier_syncthreads_vc4tile
// VC4-DAG: uses_barrier = true
// VC4-DAG: semaphores_per_block = 4 : i32
// VC4: vc4.qpu.sema
// VC4: sig = #vc4.qpu_signal<thrend>
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
