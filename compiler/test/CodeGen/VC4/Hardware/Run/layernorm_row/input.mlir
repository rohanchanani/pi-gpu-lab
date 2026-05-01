// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the layernorm_row hardware-run
// bundle.
//
// Reference semantics:
// For each row of width W <= 16:
// mean = sum(x) / W
// var = sum((x - mean)^2) / W
// out = (x - mean) * rsqrt(var + epsilon) * gamma + beta
//
// The public launcher supports scalar gamma and beta shared across all row
// elements. The test cases cover zero rows, width one, partial-width rows,
// full 16-lane rows, and more rows than active QPUs.
//
// Public API:
// layernorm_row_prepare(rt, state, max_rows, max_width)
// layernorm_row_launch(state, input, output,
// rows, width,
// epsilon, gamma, beta)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact layernorm semantics
// on the host side and tracks the one-allocation / repeated-launch
// diagnostics. This input.mlir therefore models the launchable QPU entry as
// the exact scheduled no-op kernel from the trusted qasm while preserving the
// semantic launcher ABI in vc4.launch_abi. The row-local TMU/reduction/SFU
// QPU algorithm described by the test plan is intentionally not invented
// here because it is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] input tensor buffer
// [1] output tensor buffer
// [2] rows
// [3] width
// [4] epsilon
// [5] gamma
// [6] beta

vc4.module @layernorm_row {
vc4.func @layernorm_row_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "layernorm_row_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 7 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "epsilon", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
{name = "gamma", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32},
{name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32}
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
