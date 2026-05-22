// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @metadata_generation
// CHECK-DAG: vc4.launch_abi
// CHECK-DAG: public_name = "generated_public"
// CHECK-DAG: symbol_name = "metadata_generation"
// CHECK-DAG: code_symbol = "generated_public_shader"
// CHECK-DAG: builtins = [{{.*}}name = "num_qpus"{{.*}}uniform_index = 0 : i32{{.*}}]
// CHECK-DAG: uniform_words_per_qpu = 1 : i32
// CHECK-DAG: tail_policy = "exact_multiple"
// CHECK-DAG: vc4.resource
// CHECK-DAG: schedule_mode = "cooperative_block"
// CHECK-DAG: warps_per_block_max = 3 : i32
// CHECK-DAG: uses_shared_vpm = true
// CHECK-DAG: uses_barrier = true
// CHECK-DAG: require_full_block_residency = true
// CHECK-DAG: semaphores_per_block = 4 : i32
// CHECK-DAG: vpm_rows_per_block = 4 : i32
// CHECK-DAG: vpm_bytes_per_block = 256 : i32
// CHECK: ssavc4.thread_end
vc4tile.kernel @metadata_generation attributes {
  public_name = "generated_public",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 3 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 4 : i32,
  vpm_bytes_per_block = 256 : i32
} {
  vc4tile.return
}
