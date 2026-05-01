// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the flash_attention_tiny
// hardware-run bundle.
//
// Reference semantics:
// This tiny streaming-attention reference computes the same output as
// scaled dot-product attention without materializing the score matrix:
// score[k] = scale * dot(Q[q], K[k])
// p[k] = softmax(score)[k]
// out[q, t] = sum_k p[k] * V[k, t]
//
// The trusted reference launcher uses the online update form:
// m_new = max(m, s)
// l_new = l * exp(m - m_new) + exp(s - m_new)
// acc_new = acc * (l * exp(m - m_new) / l_new) +
// V * (exp(s - m_new) / l_new)
//
// Public API:
// flash_attention_tiny_prepare(rt, state, max_q_len, max_k_len, max_d)
// flash_attention_tiny_launch(state, q, k, v, out,
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
// The committed reference launcher implements the streaming attention
// semantics on the host side and tracks the one-allocation / repeated-launch
// diagnostics. This input.mlir therefore models the launchable QPU entry as
// the exact scheduled no-op kernel from the trusted qasm while preserving the
// semantic launcher ABI in vc4.launch_abi. The streaming TMU/SFU QPU
// algorithm described by the test plan is intentionally not invented here
// because it is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] q buffer
// [1] k buffer
// [2] v buffer
// [3] out buffer
// [4] q_len
// [5] k_len
// [6] d
// [7] scale

vc4.module @flash_attention_tiny {
vc4.func @flash_attention_tiny_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "flash_attention_tiny_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 8 : i32,
args = [
{name = "q", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "k", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
{name = "v", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 2 : i32},
{name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 3 : i32},
{name = "q_len", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "k_len", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32},
{name = "d", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 6 : i32},
{name = "scale", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 7 : i32}
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
