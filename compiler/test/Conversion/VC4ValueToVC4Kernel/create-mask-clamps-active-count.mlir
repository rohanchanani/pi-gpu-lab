// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @mask_clamps(
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out",
                                           vc4value.direction = "out",
                                           vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %value = arith.constant dense<1.000000e+00> : vector<16xf32>
  vector.transfer_write %value, %out[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @mask_clamps
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: arith.constant 16 : i32
// CHECK: %[[NEG:.*]] = arith.cmpi slt,
// CHECK-SAME: %[[ZERO]] : i32
// CHECK: %[[NONNEG:.*]] = arith.select %[[NEG]], %[[ZERO]],
// CHECK: %[[ABOVE:.*]] = arith.cmpi sgt, %[[NONNEG]], {{.*}} : i32
// CHECK: %[[CLAMPED:.*]] = arith.select %[[ABOVE]], {{.*}}, %[[NONNEG]] : i32
// CHECK: vc4kernel.pred.tail %[[ZERO]], %[[CLAMPED]]
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
