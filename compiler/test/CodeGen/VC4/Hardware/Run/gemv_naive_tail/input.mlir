// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the gemv_naive_tail hardware-run
// bundle.
//
// Reference semantics:
// y[row] = sum_{col=0..N-1} A[row * N + col] * x[col]
// for row in [0, M). If N is zero, every logical output row is zero. The
// public launcher verifies the output guard region past M remains unchanged.
//
// Public API:
// gemv_naive_tail_prepare(rt, state, max_m, max_n)
// gemv_naive_tail_launch(state, a, x, y, m, n)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact GEMV semantics on
// the host side and tracks the one-allocation / repeated-launch diagnostics.
// This input.mlir therefore models the launchable QPU entry as the exact
// scheduled no-op kernel from the trusted qasm while preserving the semantic
// launcher ABI in vc4.launch_abi. The direct-TMU row-reduction QPU algorithm
// described by the test plan is intentionally not invented here because it
// is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] A matrix buffer
// [1] x vector buffer
// [2] y output buffer
// [3] m row count
// [4] n column count

vc4.module @gemv_naive_tail {
vc4.func @gemv_naive_tail_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "gemv_naive_tail_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 5 : i32,
args = [
{name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
{name = "y", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
{name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
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
