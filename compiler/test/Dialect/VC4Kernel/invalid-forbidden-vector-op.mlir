// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%x : i32) attributes {
    public_name = "bad", schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    resource = {uses_vpm = false, uses_barrier = false, require_full_block_residency = false, warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32, vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32}
  } {
    // CHECK: vector dialect operations are forbidden
    %v = vector.broadcast %x : i32 to vector<16xi32>
    vc4kernel.return
  }
}
