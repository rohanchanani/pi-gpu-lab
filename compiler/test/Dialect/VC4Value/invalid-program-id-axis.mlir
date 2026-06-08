// RUN: vc4-opt %s -verify-diagnostics

module {
  func.func @bad_program_id_negative_axis() {
    // expected-error @+1 {{axis must be 0, 1, or 2}}
    %pid = vc4value.program_id {axis = -1 : i32} : index
    return
  }

  func.func @bad_program_id_axis_too_large() {
    // expected-error @+1 {{axis must be 0, 1, or 2}}
    %pid = vc4value.program_id {axis = 3 : i32} : index
    return
  }
}
