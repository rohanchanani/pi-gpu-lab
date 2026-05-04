// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the mandelbrot_masked_loop
// hardware-run ground-truth test. The trusted runtime bundle implements the
// semantic masked-loop Mandelbrot reference; the scheduled hardware oracle for
// this corpus entry is the safe thread-end kernel shape.

vc4.module @mandelbrot_masked_loop {
vc4.func @mandelbrot_masked_loop_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "mandelbrot_masked_loop_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 10 : i32,
args = [
{name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
{name = "height", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "min_x", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
{name = "min_y", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
{name = "step_x", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32},
{name = "step_y", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32},
{name = "max_iter", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 7 : i32}
],
builtins = [
{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 8 : i32},
{name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 9 : i32}
]
}
} {
vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
}
}
