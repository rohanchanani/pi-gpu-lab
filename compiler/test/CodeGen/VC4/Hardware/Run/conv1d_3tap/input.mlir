// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the conv1d_3tap hardware
// ground-truth test.
//
// Reference semantics:
// out[i] = c0 * x[max(i - 1, 0)] +
// c1 * x[i] +
// c2 * x[min(i + 1, n - 1)]
// for i in [0, n). The boundary clamp is implemented in the QPU body, while
// the launcher pads only private scratch for inactive tail-lane safety.
//
// Physical uniform stream per active QPU:
// [0] input base bus address
// [1] output base bus address
// [2] logical element count n
// [3] c0, carried as one f32/u32 uniform word
// [4] c1, carried as one f32/u32 uniform word
// [5] c2, carried as one f32/u32 uniform word
// [6] logical_request, supplied by the launcher as a builtin suffix word
// [7] total_requests, supplied by the launcher as a builtin suffix word
//
// Work distribution:
// byte_offset = logical_request * 16 * sizeof(f32)
// i = (byte_offset / sizeof(f32)) + ELEMENT_NUMBER
// The generated launcher creates ceil(n / 16) logical requests, so each QPU
// invocation computes exactly one vector chunk. For the final partial vector,
// the active lane count is encoded into the VDW DEPTH field so only the valid
// prefix is stored.
//
// qinc-compatible setup immediates preserved from the trusted reference:
// vpm_setup(1, 1, h32(0)) = 0x00101a00
// dynamic VDW store setup base, depth in [16:23] = 0x80804000
//
// The load path is TMU0 direct memory lookup. VPM staging and VDW stores are
// protected by the global QPU mutex.

vc4.module @conv1d_3tap {
vc4.func @conv1d_3tap_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "conv1d_3tap",
tail_policy = "tail_safe",
uniform_words_per_qpu = 8 : i32,
args = [
{name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "c0", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
{name = "c1", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
{name = "c2", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32}
],
builtins = [
{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 6 : i32},
{name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 7 : i32}
]
}
} {
// Uniform stream reads.
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 6 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 7 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra8 = current byte offset, logical_request * 64.
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, small_imm = 6 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 8 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra9 = n * sizeof(f32).
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 2 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 9 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra12 = VPM row, using logical_request.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 12 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Compute remaining bytes and dynamic store depth.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = 64 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra11 = output address for this vector.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 8 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Full vector branch.
vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 80 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
}

// Tail path: ra13 = remaining_bytes / sizeof(f32).
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shr>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 40 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
}

// full_chunk:
vc4.qpu.ldi <splat32> {value = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 32 : i32}

// have_store_count:

// r2 = byte offset for this lane.
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// left_byte = max(current_byte - 4, 0).
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<max>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// right_byte = min(current_byte + 4, n_bytes - 4).
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 4 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<min>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Load left sample.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r3>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Load center sample.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Load right sample.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// out = c0 * left + c1 * center + c2 * right.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 3 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r1>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 4 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r3>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 5 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r4>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// VPM stage and VDW store.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 51 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32}
vc4.qpu.vpmvcd_setup {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<add>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 12 : i32,
  raddr_b = 0 : i32,
  add_a = #vc4.qpu_mux<r3>,
  add_b = #vc4.qpu_mux<a>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

// Dynamic-depth VDW setup and store address.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 13 : i32, raddr_b = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = -2139078656 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 32 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 12 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.vpmvcd_setup {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<add>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 0 : i32,
  raddr_b = 1 : i32,
  add_a = #vc4.qpu_mux<r2>,
  add_b = #vc4.qpu_mux<r1>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}
vc4.qpu.vpmvcd_addr {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<or>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 11 : i32,
  raddr_b = 0 : i32,
  add_a = #vc4.qpu_mux<a>,
  add_b = #vc4.qpu_mux<a>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}
vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}
vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 51 : i32, waddr_mul = 32 : i32}

// end:
vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

}
}
