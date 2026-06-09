// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @raw_scf_without_canonicalization_staged_or_invalid()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %flag = arith.cmpi ne, %c1, %c0 : i32
  scf.if %flag {
  }
  return
}

// CHECK: raw scf operation cannot lower directly to VC4Kernel
// CHECK: explicit upstream --convert-scf-to-cf
// CHECK: READY_FOR_TRITON remains NO
