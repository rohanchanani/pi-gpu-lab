// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the mandelbrot_masked_loop
// hardware-run bundle.
//
// Reference semantics:
// For each pixel (x, y), compute:
// cx = min_x + x * step_x
// cy = min_y + y * step_y
// Then iterate z = z^2 + c while |z|^2 <= 4 and iter < max_iter.
// The output is the first escape iteration count, or max_iter for points
// that do not escape. The intended hardware implementation models divergent
// SIMD behavior with per-lane active masks.
//
// Public API:
// mandelbrot_masked_loop_prepare(rt, state, max_width, max_height)
// mandelbrot_masked_loop_launch(state, output,
// width, height,
// min_x, min_y, step_x, step_y,
// max_iter)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact Mandelbrot semantics
// on the host side and tracks the one-allocation / repeated-launch
// diagnostics. This input.mlir therefore models the launchable QPU entry as
// the exact scheduled no-op kernel from the trusted qasm while preserving the
// semantic launcher ABI in vc4.launch_abi. The divergent masked-loop QPU
// algorithm described by the test plan is intentionally not invented here
// because it is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] output buffer
// [1] width
// [2] height
// [3] min_x
// [4] min_y
// [5] step_x
// [6] step_y
// [7] max_iter

vc4.module @mandelbrot_masked_loop {
vc4.func @mandelbrot_masked_loop_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "mandelbrot_masked_loop_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 8 : i32,
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
builtins = []
}
} {
// Reference qasm instruction 0: nop; thrend.
vc4.qpu.bundle {
sig = #vc4.qpu_signal<thrend>,
pm = false,
cond_add = #vc4.cond<never>,
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

// Reference qasm instruction 1: safe thread-end delay slot.
vc4.qpu.bundle {
  sig = #vc4.qpu_signal<none>,
  pm = false,
  cond_add = #vc4.cond<never>,
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

// Reference qasm instruction 2: safe thread-end delay slot.
vc4.qpu.bundle {
  sig = #vc4.qpu_signal<none>,
  pm = false,
  cond_add = #vc4.cond<never>,
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
