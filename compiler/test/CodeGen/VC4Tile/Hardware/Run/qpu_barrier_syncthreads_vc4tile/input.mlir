// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh qpu_barrier_syncthreads_vc4tile generate
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
