// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_rotate
// CHECK: ssavc4.rotate {{.*}} {amount = 3 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 0 : i32}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_rotate(%out : i32) attributes {
    public_name = "fragment_rotate",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "out", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %r3 = vc4kernel.fragment_rotate %lanes {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
    %r0 = vc4kernel.fragment_rotate %r3 {amount = 0 : i32} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %r0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
