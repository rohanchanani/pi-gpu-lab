// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_num_programs_axis attributes {
    public_name = "bad_num_programs_axis",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: 'vc4kernel.num_programs' op axis must be 0, 1, or 2
    %n = vc4kernel.num_programs {axis = -1 : i32} : i32
    vc4kernel.return
  }
}
