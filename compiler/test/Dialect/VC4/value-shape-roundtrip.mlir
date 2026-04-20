// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @value_shape {
// CHECK: vc4.func @main(%[[I:.*]]: i32, %[[F:.*]]: f32, %[[VI:.*]]: vector<16xi32>, %[[VF:.*]]: vector<16xf32>, %[[AMT:.*]]: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
// CHECK: %[[P0:.*]] = "vc4.pack"(%[[F]]) <{regfile_a_mode = #vc4.regfile_a_pack_mode<to_16a>}> : (f32) -> i32
// CHECK: %[[P1:.*]] = "vc4.pack"(%[[VF]]) <{mul_mode = #vc4.mul_pack_mode<to_8a>}> : (vector<16xf32>) -> vector<16xi32>
// CHECK: %[[U0:.*]] = "vc4.unpack"(%[[I]]) <{regfile_a_mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>}> : (i32) -> f32
// CHECK: %[[U1:.*]] = "vc4.unpack"(%[[VI]]) <{r4_mode = #vc4.r4_unpack_mode<replicate_8d>}> : (vector<16xi32>) -> vector<16xi32>
// CHECK: %[[R0:.*]] = "vc4.rotate"(%[[VI]], %[[AMT]]) : (vector<16xi32>, i32) -> vector<16xi32>
// CHECK: %[[R1:.*]] = "vc4.rotate"(%[[VF]]) <{immediate = 4 : i32}> : (vector<16xf32>) -> vector<16xf32>
// CHECK: vc4.return

vc4.module @value_shape {
  vc4.func @main(%i: i32, %f: f32, %vi: vector<16xi32>, %vf: vector<16xf32>, %amt: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %p0 = "vc4.pack"(%f) <{regfile_a_mode = #vc4.regfile_a_pack_mode<to_16a>}> : (f32) -> i32
    %p1 = "vc4.pack"(%vf) <{mul_mode = #vc4.mul_pack_mode<to_8a>}> : (vector<16xf32>) -> vector<16xi32>
    %u0 = "vc4.unpack"(%i) <{regfile_a_mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>}> : (i32) -> f32
    %u1 = "vc4.unpack"(%vi) <{r4_mode = #vc4.r4_unpack_mode<replicate_8d>}> : (vector<16xi32>) -> vector<16xi32>
    %r0 = "vc4.rotate"(%vi, %amt) : (vector<16xi32>, i32) -> vector<16xi32>
    %r1 = "vc4.rotate"(%vf) <{immediate = 4 : i32}> : (vector<16xf32>) -> vector<16xf32>
    vc4.return
  }
}
