// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @read_i32_full(%in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
                         %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @read_i32_full
// CHECK: vc4kernel.pred.full
// CHECK: arith.constant 2 : i32
// CHECK: arith.shli
// CHECK: vc4kernel.splat {{.*}} : i32 -> vector<16xi32>
// CHECK: vc4kernel.fragment_const
// CHECK-SAME: dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<add>
// CHECK: arith.constant 0 : i32
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: coherency = #vc4kernel.coherency<readonly_tmu>
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK-SAME: memory_path = #vc4kernel.memory_path<tmu_global_read>
// CHECK-SAME: : i32, vector<16xi32>, <16>, i32 -> vector<16xi32>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
