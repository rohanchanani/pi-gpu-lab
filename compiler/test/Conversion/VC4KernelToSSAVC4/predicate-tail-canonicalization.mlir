// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @tail_and_tail_canonicalized
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK-NOT: ssavc4.tmu.request
// CHECK-NOT: #vc4.add_opcode<and>
// CHECK: ssavc4.vdw.store
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tail_and_tail_canonicalized(%out : i32, %n : i32, %cap : i32) attributes {
    public_name = "tail_and_tail_canonicalized",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cap", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1711276032> : vector<16xi32>} : vector<16xi32>
    %value = vc4kernel.fragment_alu.add %tag_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tail_n = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %tail_cap = vc4kernel.pred.tail %c0, %cap : i32, i32 -> !vc4kernel.pred<16>
    %mask = vc4kernel.pred.and %tail_n, %tail_cap : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %mask {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
