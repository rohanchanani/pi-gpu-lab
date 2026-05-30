// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad", schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {uses_vpm = false, uses_barrier = false, require_full_block_residency = false, warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32, vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32}
  } {
    %true = arith.constant true
    // CHECK: raw scf operations are forbidden
    scf.if %true {
    }
    vc4kernel.return
  }
}
