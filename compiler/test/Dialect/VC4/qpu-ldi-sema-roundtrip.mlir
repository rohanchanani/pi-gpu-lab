// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @qpu_ldi_sema {
// CHECK: vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
// CHECK: vc4.qpu.ldi <splat32>
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: cond_mul = #vc4.cond<never>
// CHECK-SAME: pm = false
// CHECK-SAME: value = 42 : i32
// CHECK-SAME: waddr_add = 1 : i32
// CHECK-SAME: waddr_mul = 2 : i32
// CHECK: vc4.qpu.ldi <per_elem_u2>
// CHECK-SAME: cond_add = #vc4.cond<zs>
// CHECK-SAME: cond_mul = #vc4.cond<always>
// CHECK-SAME: pack = #vc4.mul_pack_mode<to_8a>
// CHECK-SAME: pm = true
// CHECK-SAME: set_flags
// CHECK-SAME: value = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>
// CHECK-SAME: waddr_add = 3 : i32
// CHECK-SAME: waddr_mul = 4 : i32
// CHECK-SAME: write_swap
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: cond_mul = #vc4.cond<always>
// CHECK-SAME: id = 7 : i32
// CHECK-SAME: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK-SAME: pm = false
// CHECK-SAME: waddr_add = 5 : i32
// CHECK-SAME: waddr_mul = 6 : i32

vc4.module @qpu_ldi_sema {
  vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.ldi <splat32> {
      value = 42 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 1 : i32,
      waddr_mul = 2 : i32
    }

    vc4.qpu.ldi <per_elem_u2> {
      value = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>,
      pm = true,
      pack = #vc4.mul_pack_mode<to_8a>,
      cond_add = #vc4.cond<zs>,
      cond_mul = #vc4.cond<always>,
      set_flags,
      write_swap,
      waddr_add = 3 : i32,
      waddr_mul = 4 : i32
    }

    vc4.qpu.sema <acquire> {
      id = 7 : i32,
      pm = false,
      pack = #vc4.regfile_a_pack_mode<to_16a>,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 5 : i32,
      waddr_mul = 6 : i32
    }
  }
}
