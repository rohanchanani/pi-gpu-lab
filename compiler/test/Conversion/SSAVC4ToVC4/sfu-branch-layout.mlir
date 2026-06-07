// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @sfu_branch_layout
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, immediate = 112 : i32
// CHECK: waddr_add = 52 : i32
// CHECK: waddr_add = 32 : i32
// CHECK: waddr_add = 32 : i32
// CHECK: add_a = #vc4.qpu_mux<r4>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, immediate = 72 : i32
ssavc4.module @sfu_branch_layout {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %one, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^skip, ^sfu_path {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags
  ^sfu_path:
    %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
    %sfu = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    ssavc4.br ^done
  ^skip:
    %fallback = ssavc4.load_imm <splat32> {value = 2.000000e+00 : f32} : vector<16xf32>
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
