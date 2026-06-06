// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_bitcast_lowering
// CHECK: ssavc4.mov {{.*}} : vector<16xi32> -> vector<16xf32>
// CHECK: ssavc4.mov {{.*}} : vector<16xf32> -> vector<16xi32>
// CHECK-NOT: #vc4.add_opcode<itof>
// CHECK-NOT: #vc4.add_opcode<ftoi>
// CHECK-NOT: ssavc4.alu.add
// CHECK-NOT: vc4.qpu
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_bitcast_lowering(%i : i32, %f : f32) attributes {
    public_name = "fragment_bitcast_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "i", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "f", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %iv = vc4kernel.splat %i : i32 -> vector<16xi32>
    %fv = vc4kernel.splat %f : f32 -> vector<16xf32>
    %as_f = vc4kernel.fragment_bitcast %iv : vector<16xi32> -> vector<16xf32>
    %as_i = vc4kernel.fragment_bitcast %fv : vector<16xf32> -> vector<16xi32>
    %back_i = vc4kernel.fragment_bitcast %as_f : vector<16xf32> -> vector<16xi32>
    %back_f = vc4kernel.fragment_bitcast %as_i : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}
