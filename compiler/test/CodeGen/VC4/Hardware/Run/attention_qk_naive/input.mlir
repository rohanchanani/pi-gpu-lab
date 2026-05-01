// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the attention_qk_naive
// hardware-run bundle.
//
// Reference semantics:
// Given row-major Q[Q_LEN, D] and K[K_LEN, D], compute:
// scores[q, k] = scale * sum_{t=0..D-1}(Q[q, t] * K[k, t])
//
// The first reference version keeps K_LEN <= 16 and D <= 16 so a future QPU
// implementation can naturally map one score row to a single VC4 vector.
//
// Public API:
// attention_qk_naive_prepare(rt, state, max_q_len, max_k_len, max_d)
// attention_qk_naive_launch(state, q, k, scores,
// q_len, k_len, d, scale)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact Q*K^T score
// semantics on the host side and tracks the one-allocation / repeated-launch
// diagnostics. This input.mlir therefore models the launchable QPU entry as
// the exact scheduled no-op kernel from the trusted qasm while preserving the
// semantic launcher ABI in vc4.launch_abi. The direct-TMU QK dot-product
// QPU algorithm described by the test plan is intentionally not invented
// here because it is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] q buffer
// [1] k buffer
// [2] scores buffer
// [3] q_len
// [4] k_len
// [5] d
// [6] scale

vc4.module @attention_qk_naive {
vc4.func @attention_qk_naive_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "attention_qk_naive_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 7 : i32,
args = [
{name = "q", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "k", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
{name = "scores", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
{name = "q_len", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "k_len", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "d", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32},
{name = "scale", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32}
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
