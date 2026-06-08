// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @constants_and_splats(%x: i32 {vc4value.arg_name = "x"},
                                %alpha: f32 {vc4value.arg_name = "alpha"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %ci = arith.constant 7 : index
  %c32 = arith.constant 11 : i32
  %cf = arith.constant 2.500000e+00 : f32
  %vi = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  %v32 = arith.constant dense<42> : vector<16xi32>
  %vf = arith.constant dense<1.500000e+00> : vector<16xf32>
  %sx = vector.broadcast %x : i32 to vector<16xi32>
  %si = vector.broadcast %ci : index to vector<16xindex>
  %sf = vector.broadcast %alpha : f32 to vector<16xf32>
  %sum = arith.addi %v32, %sx : vector<16xi32>
  %idx_sum = arith.addi %vi, %si : vector<16xindex>
  %fsum = arith.addf %vf, %sf : vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @constants_and_splats
// CHECK: arith.constant 7 : i32
// CHECK: arith.constant 11 : i32
// CHECK: arith.constant 2.500000e+00 : f32
// CHECK: vc4kernel.fragment_const
// CHECK-SAME: dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
// CHECK: vc4kernel.fragment_const
// CHECK-SAME: dense<42> : vector<16xi32>
// CHECK: vc4kernel.fragment_const
// CHECK-SAME: dense<1.500000e+00> : vector<16xf32>
// CHECK: vc4kernel.splat {{.*}} : i32 -> vector<16xi32>
// CHECK: vc4kernel.splat {{.*}} : i32 -> vector<16xi32>
// CHECK: vc4kernel.splat {{.*}} : f32 -> vector<16xf32>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
