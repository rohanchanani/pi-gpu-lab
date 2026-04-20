// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @sfu_ops {
// CHECK: vc4.func @main(%[[I:.*]]: i32, %[[F:.*]]: f32, %[[VI:.*]]: vector<16xi32>, %[[VF:.*]]: vector<16xf32>) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
// CHECK: vc4.sfu.issue <recip> %[[F]] : f32
// CHECK: vc4.sfu.issue <recipsqrt> %[[VF]] : vector<16xf32>
// CHECK: vc4.sfu.issue <exp> %[[I]] : i32
// CHECK: vc4.sfu.issue <log> %[[VI]] : vector<16xi32>
// CHECK: %[[R0:.*]] = vc4.sfu.read : f32
// CHECK: %[[R1:.*]] = vc4.sfu.read : vector<16xf32>
// CHECK: vc4.return

vc4.module @sfu_ops {
  vc4.func @main(%i: i32, %f: f32, %vi: vector<16xi32>, %vf: vector<16xf32>) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    vc4.sfu.issue <recip> %f : f32
    vc4.sfu.issue <recipsqrt> %vf : vector<16xf32>
    vc4.sfu.issue <exp> %i : i32
    vc4.sfu.issue <log> %vi : vector<16xi32>
    %r0 = vc4.sfu.read : f32
    %r1 = vc4.sfu.read : vector<16xf32>
    vc4.return
  }
}
