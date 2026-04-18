// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @alu_ops {
// CHECK: vc4.func @main(%[[A:.*]]: i32, %[[B:.*]]: i32, %[[FA:.*]]: f32, %[[FB:.*]]: f32, %[[VI:.*]]: vector<16xi32>, %[[VF:.*]]: vector<16xf32>) -> i32 attributes {form = 0 : i32, threading = 0 : i32} {
// CHECK: %[[ADD:.*]] = vc4.alu.add <add> %[[A]], %[[B]] {cond = #vc4.cond<always>} : (i32, i32) -> i32
// CHECK: %[[ITOF:.*]] = vc4.alu.add <itof> %[[A]] {cond = #vc4.cond<always>} : (i32) -> f32
// CHECK: %[[FADD:.*]] = vc4.alu.add <fadd> %[[FA]], %[[FB]] {cond = #vc4.cond<zs>, set_flags} : (f32, f32) -> f32
// CHECK: %[[BITNOT:.*]] = vc4.alu.add <not> %[[VI]] {cond = #vc4.cond<always>} : (vector<16xi32>) -> vector<16xi32>
// CHECK: %[[FMUL:.*]] = vc4.alu.mul <fmul> %[[FA]], %[[FB]] {cond = #vc4.cond<always>} : (f32, f32) -> f32
// CHECK: %[[MUL24:.*]] = vc4.alu.mul <mul24> %[[VI]], %[[VI]] {cond = #vc4.cond<nc>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
// CHECK: %[[LDI0:.*]] = vc4.load_imm {mode = #vc4.load_imm_mode<splat32>, value = 1065353216 : i32} : f32
// CHECK: %[[LDI1:.*]] = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_i2>, value = array<i32: -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1>} : vector<16xi32>
// CHECK: %[[LDI2:.*]] = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_u2>, value = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>} : vector<16xi32>
// CHECK: vc4.return %[[ADD]] : i32

vc4.module @alu_ops {
  vc4.func @main(%a: i32, %b: i32, %fa: f32, %fb: f32, %vi: vector<16xi32>, %vf: vector<16xf32>) -> i32 attributes {threading = 0 : i32, form = 0 : i32} {
    %add = vc4.alu.add <add> %a, %b {cond = #vc4.cond<always>} : (i32, i32) -> i32
    %itof = vc4.alu.add <itof> %a {cond = #vc4.cond<always>} : (i32) -> f32
    %fadd = vc4.alu.add <fadd> %fa, %fb {cond = #vc4.cond<zs>, set_flags} : (f32, f32) -> f32
    %bitnot = vc4.alu.add <not> %vi {cond = #vc4.cond<always>} : (vector<16xi32>) -> vector<16xi32>
    %fmul = vc4.alu.mul <fmul> %fa, %fb {cond = #vc4.cond<always>} : (f32, f32) -> f32
    %mul24 = vc4.alu.mul <mul24> %vi, %vi {cond = #vc4.cond<nc>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %ldi0 = vc4.load_imm {mode = #vc4.load_imm_mode<splat32>, value = 1065353216 : i32} : f32
    %ldi1 = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_i2>, value = array<i32: -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1>} : vector<16xi32>
    %ldi2 = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_u2>, value = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>} : vector<16xi32>
    vc4.return %add : i32
  }
}
