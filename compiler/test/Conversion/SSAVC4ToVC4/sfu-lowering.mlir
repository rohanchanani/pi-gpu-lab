// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @sfu_lowering
// CHECK: vc4.qpu.bundle {{.*}}add_a = #vc4.qpu_mux<a>{{.*}}add_b = #vc4.qpu_mux<a>{{.*}}waddr_add = 52 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-NOT: #vc4.qpu_mux<r4>
// CHECK: waddr_add = 32 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-NOT: #vc4.qpu_mux<r4>
// CHECK: waddr_add = 32 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: add_a = #vc4.qpu_mux<r4>
// CHECK-SAME: add_b = #vc4.qpu_mux<r4>
// CHECK: waddr_add = 53 : i32
// CHECK: waddr_add = 54 : i32
// CHECK: waddr_add = 55 : i32
// CHECK-NOT: ssavc4.
ssavc4.module @sfu_lowering {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
    %recip = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %rsqrt = ssavc4.sfu %recip {kind = #ssavc4.sfu_kind<rsqrt>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    %exp = ssavc4.sfu %rsqrt {kind = #ssavc4.sfu_kind<exp>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %log = ssavc4.sfu %exp {kind = #ssavc4.sfu_kind<log>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    ssavc4.thread_end
  }
}
