// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @uniforms {
// CHECK: vc4.func @main(%[[SEED:.*]]: vector<16xf32>, %[[OFS:.*]]: i32) -> i32 attributes {form = 0 : i32, threading = 0 : i32} {
// CHECK: %[[U0:.*]] = vc4.uniform.read : i32
// CHECK: vc4.uniform.seek %[[OFS]] : i32
// CHECK: vc4.uniform.seek %[[OFS]] {relative = true} : i32
// CHECK: %[[MOV:.*]] = vc4.mov %[[SEED]] : vector<16xf32>
// CHECK: %[[READ:.*]] = vc4.read %[[MOV]] : vector<16xf32>
// CHECK: vc4.return %[[U0]] : i32

vc4.module @uniforms {
  vc4.func @main(%seed: vector<16xf32>, %ofs: i32) -> i32 attributes {threading = 0 : i32, form = 0 : i32} {
    %u0 = vc4.uniform.read : i32
    vc4.uniform.seek %ofs : i32
    vc4.uniform.seek %ofs {relative = true} : i32
    %m = vc4.mov %seed : vector<16xf32>
    %r = vc4.read %m : vector<16xf32>
    vc4.return %u0 : i32
  }
}
