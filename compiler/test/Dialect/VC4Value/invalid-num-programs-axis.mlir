// RUN: vc4-opt %s -verify-diagnostics

module {
  func.func @bad_num_programs_negative_axis() {
    // expected-error @+1 {{axis must be 0, 1, or 2}}
    %n = vc4value.num_programs {axis = -1 : i32} : index
    return
  }

  func.func @bad_num_programs_axis_too_large() {
    // expected-error @+1 {{axis must be 0, 1, or 2}}
    %n = vc4value.num_programs {axis = 3 : i32} : index
    return
  }
}
