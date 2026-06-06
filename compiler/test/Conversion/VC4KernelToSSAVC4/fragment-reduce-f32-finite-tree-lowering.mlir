// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_reduce_f32_finite_tree
// CHECK: ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xf32>
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.rotate {{.*}} {amount = 8 : i32}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fadd>}
// CHECK: ssavc4.rotate {{.*}} {amount = 4 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 2 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 1 : i32}
// CHECK: ssavc4.load_imm <splat32> {value = 2139095040 : i32} : vector<16xf32>
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fmin>}
// CHECK: ssavc4.load_imm <splat32> {value = -8388608 : i32} : vector<16xf32>
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fmax>}
// CHECK-NOT: vc4.qpu.
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_reduce_f32_finite_tree(%out : i32, %value : f32) attributes {
    public_name = "fragment_reduce_f32_finite_tree",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "value", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %c0 = arith.constant 0 : i32
    %c9 = arith.constant 9 : i32
    %tail = vc4kernel.pred.tail %c0, %c9 : i32, i32 -> !vc4kernel.pred<16>
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %values = vc4kernel.splat %value : f32 -> vector<16xf32>
    %sum = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %min = vc4kernel.fragment_reduce %values, %empty {kind = #vc4kernel.reduce<fmin>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %max = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<fmax>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %selected = vc4kernel.fragment_select %full, %sum, %min : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %outv = vc4kernel.fragment_alu.add %selected, %max {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %outv, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
