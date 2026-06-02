// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  // CHECK: only {{vc4kernel[.]kernel}} operations may appear at module top level
  %pid = vc4kernel.program_id {axis = 0 : i32} : i32
}
