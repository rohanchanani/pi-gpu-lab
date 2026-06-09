// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @cf_counted_loop_scalar_block_args(%n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^loop(%c0 : index)

^loop(%i: index):
  %done = arith.cmpi uge, %i, %n : index
  cf.cond_br %done, ^exit, ^body(%i : index)

^body(%body_i: index):
  %next = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next : index)

^exit:
  return
}

// CHECK-LABEL: vc4kernel.kernel @cf_counted_loop_scalar_block_args
// CHECK: cf.br
// CHECK: ^{{.*}}(%{{.*}}: i32)
// CHECK: cf.cond_br
// CHECK: arith.addi
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
