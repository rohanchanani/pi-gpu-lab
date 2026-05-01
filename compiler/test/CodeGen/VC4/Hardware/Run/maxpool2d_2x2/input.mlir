// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the maxpool2d_2x2 hardware-run
// bundle.
//
// Reference semantics:
// The public launcher computes a row-major f32 2x2 max-pooling operation.
// For input shape height x width:
// out_w = ceil(width / 2)
// out_h = ceil(height / 2)
// Each output element is the maximum of valid in-bounds elements in the 2x2
// input window starting at (2 * oy, 2 * ox). Odd-width and odd-height edge
// windows use only valid input elements.
//
// Public API:
// maxpool2d_2x2_prepare(rt, state, max_width, max_height)
// maxpool2d_2x2_launch(state, input, output, width, height)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact max-pooling
// semantics on the host side and tracks the one-allocation / repeated-launch
// diagnostics. This input.mlir therefore models the launchable QPU entry as
// the exact scheduled no-op kernel from the trusted qasm while preserving the
// semantic launcher ABI in vc4.launch_abi. The direct-TMU max-pool QPU
// algorithm described by the test plan is intentionally not invented here
// because it is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] input image buffer
// [1] output pooled image buffer
// [2] width
// [3] height

vc4.module @maxpool2d_2x2 {
vc4.func @maxpool2d_2x2_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "maxpool2d_2x2_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 4 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "height", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
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
