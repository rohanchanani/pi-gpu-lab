// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @scf_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %flag = arith.cmpi ult, %c0, %c1 : i32
  scf.if %flag {
  }
  return
}

// CHECK: control-flow operation is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
