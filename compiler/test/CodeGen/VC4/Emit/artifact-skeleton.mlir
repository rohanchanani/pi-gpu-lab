// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --input-file=%t.bundle/manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.bundle/kernel.qasm --check-prefix=QASM
// RUN: FileCheck %s --input-file=%t.bundle/kernel_launch.h --check-prefix=HEADER
// RUN: FileCheck %s --input-file=%t.bundle/kernel_launch.c --check-prefix=LAUNCHC

vc4.module @emit_artifact_skeleton {
 vc4.func @minimal_thrend_kernel() attributes {
 domain = #vc4.execution_domain<qpu>,
 form = #vc4.function_form<scheduled>,
 kernel,
 threading = #vc4.threading_mode<single>,
 "vc4.launch_abi" = {
 public_name = "minimal_thrend_launch",
 tail_policy = "exact_multiple",
 uniform_words_per_qpu = 2 : i32,
 args = [],
 builtins = [
 {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
 {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
 ]
 }
 } {
 vc4.qpu.bundle {
 sig = #vc4.qpu_signal<thrend>,
 pm = false,
 cond_add = #vc4.cond<always>,
 cond_mul = #vc4.cond<never>,
 waddr_add = 32 : i32,
 waddr_mul = 33 : i32,
 op_add = #vc4.add_opcode<nop>,
 op_mul = #vc4.mul_opcode<nop>,
 raddr_a = 0 : i32,
 raddr_b = 1 : i32,
 add_a = #vc4.qpu_mux<a>,
 add_b = #vc4.qpu_mux<b>,
 mul_a = #vc4.qpu_mux<r0>,
 mul_b = #vc4.qpu_mux<r1>
 }

 vc4.qpu.bundle {
 sig = #vc4.qpu_signal<none>,
 pm = false,
 cond_add = #vc4.cond<always>,
 cond_mul = #vc4.cond<never>,
 waddr_add = 32 : i32,
 waddr_mul = 33 : i32,
 op_add = #vc4.add_opcode<nop>,
 op_mul = #vc4.mul_opcode<nop>,
 raddr_a = 0 : i32,
 raddr_b = 1 : i32,
 add_a = #vc4.qpu_mux<a>,
 add_b = #vc4.qpu_mux<b>,
 mul_a = #vc4.qpu_mux<r0>,
 mul_b = #vc4.qpu_mux<r1>
 }

 vc4.qpu.bundle {
 sig = #vc4.qpu_signal<none>,
 pm = false,
 cond_add = #vc4.cond<always>,
 cond_mul = #vc4.cond<never>,
 waddr_add = 32 : i32,
 waddr_mul = 33 : i32,
 op_add = #vc4.add_opcode<nop>,
 op_mul = #vc4.mul_opcode<nop>,
 raddr_a = 0 : i32,
 raddr_b = 1 : i32,
 add_a = #vc4.qpu_mux<a>,
 add_b = #vc4.qpu_mux<b>,
 mul_a = #vc4.qpu_mux<r0>,
 mul_b = #vc4.qpu_mux<r1>
 }
 }
}

// MANIFEST: "kernel": "minimal_thrend_kernel"
// MANIFEST: "public_name": "minimal_thrend_launch"
// MANIFEST: "placeholder": true
// QASM: ; vc4-codegen placeholder qasm
// QASM: ; kernel: minimal_thrend_kernel
// HEADER: int minimal_thrend_launch(void);
// LAUNCHC: int minimal_thrend_launch(void) {
