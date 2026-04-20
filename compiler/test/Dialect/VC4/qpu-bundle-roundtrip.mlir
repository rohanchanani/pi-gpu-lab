// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @qpu_bundle {
// CHECK: vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
// CHECK: vc4.qpu.bundle
// CHECK-SAME: add_a = #vc4.qpu_mux<a>
// CHECK-SAME: add_b = #vc4.qpu_mux<b>
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: cond_mul = #vc4.cond<never>
// CHECK-SAME: mul_a = #vc4.qpu_mux<r0>
// CHECK-SAME: mul_b = #vc4.qpu_mux<r1>
// CHECK-SAME: op_add = #vc4.add_opcode<add>
// CHECK-SAME: op_mul = #vc4.mul_opcode<nop>
// CHECK-SAME: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK-SAME: pm = false
// CHECK-SAME: raddr_a = 32 : i32
// CHECK-SAME: raddr_b = 35 : i32
// CHECK-SAME: sig = #vc4.qpu_signal<none>
// CHECK-SAME: unpack = #vc4.regfile_a_unpack_mode<f16a_or_i16a>
// CHECK-SAME: waddr_add = 1 : i32
// CHECK-SAME: waddr_mul = 2 : i32
// CHECK: vc4.qpu.bundle
// CHECK-SAME: add_a = #vc4.qpu_mux<a>
// CHECK-SAME: add_b = #vc4.qpu_mux<b>
// CHECK-SAME: cond_add = #vc4.cond<zs>
// CHECK-SAME: cond_mul = #vc4.cond<always>
// CHECK-SAME: mul_a = #vc4.qpu_mux<r4>
// CHECK-SAME: mul_b = #vc4.qpu_mux<b>
// CHECK-SAME: op_add = #vc4.add_opcode<sub>
// CHECK-SAME: op_mul = #vc4.mul_opcode<fmul>
// CHECK-SAME: pack = #vc4.mul_pack_mode<to_8a>
// CHECK-SAME: pm = true
// CHECK-SAME: raddr_a = 4 : i32
// CHECK-SAME: set_flags
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 49 : i32
// CHECK-SAME: unpack = #vc4.r4_unpack_mode<f16a>
// CHECK-SAME: waddr_add = 5 : i32
// CHECK-SAME: waddr_mul = 6 : i32
// CHECK-SAME: write_swap

vc4.module @qpu_bundle {
  vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      unpack = #vc4.regfile_a_unpack_mode<f16a_or_i16a>,
      pm = false,
      pack = #vc4.regfile_a_pack_mode<to_16a>,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 1 : i32,
      waddr_mul = 2 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 32 : i32,
      raddr_b = 35 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<small_imm>,
      unpack = #vc4.r4_unpack_mode<f16a>,
      pm = true,
      pack = #vc4.mul_pack_mode<to_8a>,
      cond_add = #vc4.cond<zs>,
      cond_mul = #vc4.cond<always>,
      set_flags,
      write_swap,
      waddr_add = 5 : i32,
      waddr_mul = 6 : i32,
      op_add = #vc4.add_opcode<sub>,
      op_mul = #vc4.mul_opcode<fmul>,
      raddr_a = 4 : i32,
      small_imm = 49 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r4>,
      mul_b = #vc4.qpu_mux<b>
    }
  }
}
