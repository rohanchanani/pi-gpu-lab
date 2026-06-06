// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @tail_store
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<sub>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.vdw.store
// CHECK: ssavc4.vdw.store
// CHECK-LABEL: ssavc4.func @empty_store
// CHECK-NOT: ssavc4.cond_br
// CHECK-NOT: ssavc4.vdw.store
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tail_store(%out : i32, %base_index : i32, %limit : i32) attributes {
    public_name = "tail_store",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "base_index", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %base_bytes = arith.shli %base_index, %c2 : i32
    %tail = vc4kernel.pred.tail %base_index, %limit : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }

  vc4kernel.kernel @empty_store(%out : i32) attributes {
    public_name = "empty_store",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "out", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %empty {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
