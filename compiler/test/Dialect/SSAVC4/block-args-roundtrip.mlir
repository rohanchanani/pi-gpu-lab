// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @block_args_roundtrip
// CHECK: ssavc4.br ^{{.*}}
// CHECK: ssavc4.br ^{{.*}}(%{{.*}} : i32)
// CHECK: ^{{.*}}(%{{.*}}: i32):
// CHECK: ssavc4.br ^{{.*}}(%{{.*}}, %{{.*}} : i32, vector<16xf32>)
// CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xf32>):
// CHECK: ssavc4.cond_br %{{.*}}, ^{{.*}}, ^{{.*}} {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
// CHECK: ssavc4.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : i32), ^{{.*}}(%{{.*}} : i32) {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
ssavc4.module @block_args_roundtrip {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %f = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : f32
    %vf = ssavc4.splat %f : f32 -> vector<16xf32>
    ssavc4.br ^no_arg
  ^no_arg:
    ssavc4.br ^one_arg(%zero : i32)
  ^one_arg(%arg0: i32):
    ssavc4.br ^mixed_args(%arg0, %vf : i32, vector<16xf32>)
  ^mixed_args(%mix0: i32, %mix1: vector<16xf32>):
    %flags0 = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags0, ^true_no_args, ^false_no_args {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
  ^true_no_args:
    ssavc4.br ^cond_args
  ^false_no_args:
    ssavc4.br ^cond_args
  ^cond_args:
    %flags1 = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags1, ^then_arg(%mix0 : i32), ^else_arg(%zero : i32) {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^then_arg(%then_value: i32):
    ssavc4.br ^done
  ^else_arg(%else_value: i32):
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
