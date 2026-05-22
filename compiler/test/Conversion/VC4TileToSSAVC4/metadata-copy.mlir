// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @metadata_copy
// CHECK-SAME: vc4.launch_abi = {{.*}}public_name = "explicit_public"
// CHECK-SAME: vc4.resource = {{.*}}schedule_mode = "cooperative_block"
// CHECK-SAME: warps_per_block_max = 2 : i32
vc4tile.kernel @metadata_copy attributes {
  launch_abi = {
    public_name = "explicit_public",
    code_symbol = "explicit_public_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [],
    builtins = []
  },
  resource = {
    schedule_mode = "cooperative_block",
    warps_per_block_max = 2 : i32,
    uses_shared_vpm = false,
    uses_barrier = false,
    require_full_block_residency = false,
    semaphores_per_block = 0 : i32,
    vpm_rows_per_block = 0 : i32,
    vpm_bytes_per_block = 0 : i32
  },
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32
} {
  vc4tile.return
}
