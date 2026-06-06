// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// Signed edge cases for P3 hardware coverage: INT_MIN, INT_MAX, -1, 0, 1.
// P3b conversion uses sign-bias lowering: x signed-cmp y becomes
// (x xor 0x80000000) unsigned-cmp (y xor 0x80000000).

// CHECK-LABEL: ssavc4.func @fragment_cmp_i32_signed_lowering
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_cmp_i32_signed_lowering(%out : i32, %lhs : i32, %rhs : i32) attributes {
    public_name = "fragment_cmp_i32_signed_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "lhs", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rhs", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lhs_v = vc4kernel.splat %lhs : i32 -> vector<16xi32>
    %rhs_v = vc4kernel.splat %rhs : i32 -> vector<16xi32>
    %one = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %two = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %three = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %four = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %slt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<slt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_slt = vc4kernel.fragment_select %slt, %one, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sle = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<sle>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_sle = vc4kernel.fragment_select %sle, %two, %sel_slt : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sgt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<sgt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_sgt = vc4kernel.fragment_select %sgt, %three, %sel_sle : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sge = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<sge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel_sge = vc4kernel.fragment_select %sge, %four, %sel_sgt : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offsets, %sel_sge, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
