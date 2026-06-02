// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_program_id_axis attributes {
    public_name = "bad_program_id_axis",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: 'vc4kernel.program_id' op axis must be 0, 1, or 2
    %pid = vc4kernel.program_id {axis = 3 : i32} : i32
    vc4kernel.return
  }
}
