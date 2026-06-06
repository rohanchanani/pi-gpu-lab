// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_f32_arith
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fadd>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fsub>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<fmul>}
// CHECK-NOT: #vc4.mul_opcode<mul24>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_f32_arith(%out : i32, %alpha : f32, %beta : f32) attributes {
    public_name = "fragment_f32_arith",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %a = vc4kernel.splat %alpha : f32 -> vector<16xf32>
    %b = vc4kernel.splat %beta : f32 -> vector<16xf32>
    %sum = vc4kernel.fragment_alu.add %a, %b {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %diff = vc4kernel.fragment_alu.add %sum, %a {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %prod = vc4kernel.fragment_alu.mul %diff, %b {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %prod, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
