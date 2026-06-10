// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt %s --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

module {
  tt.func public @axis_out_of_range(%out: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %pid = "tt.get_program_id"() {axis = 3 : i32} : () -> i32
    tt.return
  }
}

// CHECK: 'tt.get_program_id' op attribute 'axis' failed to satisfy constraint
// CHECK: allowed 32-bit signless integer cases: 0, 1, 2
