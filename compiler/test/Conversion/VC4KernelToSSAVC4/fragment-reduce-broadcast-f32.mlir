// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_reduce_broadcast_f32
// CHECK: ssavc4.rotate {{.*}} {amount = 8 : i32}
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<fadd>
// CHECK: ssavc4.rotate {{.*}} {amount = 4 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 2 : i32}
// CHECK: ssavc4.rotate {{.*}} {amount = 1 : i32}
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_reduce_broadcast_f32(%out : i32, %value : f32) attributes {
    public_name = "fragment_reduce_broadcast_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "value", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %values = vc4kernel.splat %value : f32 -> vector<16xf32>
    %sum = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %sum, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
