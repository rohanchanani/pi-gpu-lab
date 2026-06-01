// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @too_many_rows attributes {
    public_name = "too_many_rows",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %tile0 = vc4kernel.vpm_alloc {rows = 64 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: computed VPM row requirement exceeds 64 rows
    %tile1 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.return
  }
}
