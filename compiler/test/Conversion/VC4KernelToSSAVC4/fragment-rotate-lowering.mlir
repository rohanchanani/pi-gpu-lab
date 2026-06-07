// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_rotate
// CHECK: ssavc4.rotate {{.*}} {amount = 3 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 0 : i32}
// CHECK: ssavc4.rotate {{.*}}, {{.*}} : vector<16xi32>, i32 -> vector<16xi32>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_rotate(%out : i32, %amount : i32) attributes {
    public_name = "fragment_rotate",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %r3 = vc4kernel.fragment_rotate %lanes {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
    %r0 = vc4kernel.fragment_rotate %r3 {amount = 0 : i32} : vector<16xi32> -> vector<16xi32>
    %rd = vc4kernel.fragment_rotate %r0, %amount : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %rd, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
