// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @sfu_tmu_r4_ordering
// CHECK: waddr_add = 56 : i32
// CHECK: sig = #vc4.qpu_signal<ldtmu0>
// CHECK: add_a = #vc4.qpu_mux<r4>
// CHECK-SAME: add_b = #vc4.qpu_mux<r4>
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
ssavc4.module @sfu_tmu_r4_ordering {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %tok = ssavc4.tmu.request %addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %value = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %as_f32 = ssavc4.mov %value : vector<16xi32> -> vector<16xf32>
    %sfu = ssavc4.sfu %as_f32 {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    ssavc4.thread_end
  }
}
