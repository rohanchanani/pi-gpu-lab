// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  func.func @bad_program_id_result_type() {
    // expected-error @+1 {{custom op 'vc4value.program_id' invalid kind of type specified: expected builtin.index, but found 'i32'}}
    %pid = vc4value.program_id {axis = 0 : i32} : i32
    return
  }
}

// -----

module {
  func.func @bad_num_programs_result_type() {
    // expected-error @+1 {{custom op 'vc4value.num_programs' invalid kind of type specified: expected builtin.index, but found 'i32'}}
    %n = vc4value.num_programs {axis = 0 : i32} : i32
    return
  }
}
