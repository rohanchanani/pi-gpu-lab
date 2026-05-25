// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the softmax_row hardware
// ground-truth test.
//
// Reference semantics, for each row and active lane i < width:
// mean = sum(x[i]) / width
// var = sum((x[i] - mean) * (x[i] - mean)) / width
// out[i] = (x[i] - mean) * rsqrt(var + epsilon) * gamma + beta
//
// The checked-in qasm for this test is intentionally minimal; the semantic
// source of truth is the launcher/harness contract. This scheduled body is a
// conservative one-QPU-warp-per-row implementation for width <= 16. Rows are
// distributed round-robin by qpu_id, with one 16-lane vector per row.
//
// Physical uniform stream per active QPU:
// [0] input scratch base bus address
// [1] output scratch base bus address
// [2] rows
// [3] width
// [4] padded_width, equal to round_up(width, 16) in the scratch layout
// [5] epsilon, carried as one f32/u32 uniform word
// [6] gamma, carried as one f32/u32 uniform word
// [7] beta, carried as one f32/u32 uniform word
// [8] qpu_id, supplied by the launcher as a builtin suffix word
// [9] num_qpus, supplied by the launcher as a builtin suffix word
//
// The launcher pads inactive input lanes with zero, and this body also masks
// lane contributions for the variance pass before reduction. The VDW store
// uses dynamic DEPTH = width so only the active row prefix is written.
//
// SFU usage:
// width is converted to f32 and sent through SFU_RECIP for 1/width.
// var + epsilon is sent through SFU_RECIPSQRT for inv_std.
// Two safe spacer instructions separate each SFU issue from the r4 read.
//
// qinc-compatible setup immediates used by related hardware-run kernels:
// vpm_setup(1, 1, h32(0)) = 0x00101a00
// dynamic VDW store setup base = 0x80804000

vc4.module @softmax_row {
vc4.func @softmax_row_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "softmax_row",
tail_policy = "tail_safe",
uniform_words_per_qpu = 10 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "padded_width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "epsilon", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32},
{name = "gamma", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32},
{name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 7 : i32}
],
builtins = [
{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 8 : i32},
{name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 9 : i32}
]
}
} {
// Uniform reads.
// ra0 = input base
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra1 = output base
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra2 = rows
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 2 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra3 = width
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 3 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra4 = padded_width
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 4 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra5 = epsilon
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 5 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra6 = gamma
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 6 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra7 = beta
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 7 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra8 = qpu_id
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 8 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// ra9 = num_qpus
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 9 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
// Mirror scalar uniforms used through the B read port.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 2 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 2 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 3 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 4 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 4 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 5 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 5 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 9 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 9 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra10 = current row, initialized to qpu_id.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 10 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// row_check: if current row >= rows, exit.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 31 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 10 : i32, raddr_b = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 880 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
}

// row_byte_base = current_row * padded_width * sizeof(float), in ra18.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<mul24>, raddr_a = 10 : i32, raddr_b = 4 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<a>, mul_b = #vc4.qpu_mux<b>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 18 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 31 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Direct TMU load of one padded row vector.
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 18 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 56 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<ldtmu0>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// ra11 = active x values, with lanes >= width forced to zero.
vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 33 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, raddr_b = 3 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<cs>, cond_mul = #vc4.cond<never>, waddr_add = 11 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Evaluate SFU exp(active_x), then mask inactive lanes to zero in ra14.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 54 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 14 : i32, waddr_mul = 33 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, raddr_b = 3 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<cs>, cond_mul = #vc4.cond<never>, waddr_add = 14 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// inv_width = SFU_RECIP((float)width), in ra12.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<itof>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 52 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 12 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// mean = sum * inv_width, in ra13.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 12 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 13 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 13 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 13 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// delta = active_x - mean, then mask inactive lanes to zero in ra15.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fsub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 11 : i32, raddr_b = 13 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 15 : i32, waddr_mul = 33 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 33 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 38 : i32, raddr_b = 3 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<cs>, cond_mul = #vc4.cond<never>, waddr_add = 15 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, write_swap, waddr_add = 15 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 15 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Reduce active exp values in r0 to a replicated row sum.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 14 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 31 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r0>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 56 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 52 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 50 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, small_imm = 49 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<fadd>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// inv_sum = SFU_RECIP(sum(exp(x))), in ra17.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 52 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 16 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// SFU result spacers, then capture inv_sum.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 16 : i32, raddr_b = 5 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 53 : i32, waddr_mul = 31 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 17 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r4>, add_b = #vc4.qpu_mux<r4>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// result = exp(x) * inv_sum, left in r0.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 14 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<always>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 17 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 31 : i32, waddr_mul = 34 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 6 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r2>, mul_b = #vc4.qpu_mux<a>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r2>, add_b = #vc4.qpu_mux<r2>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

// Mutex-protected VPM stage and dynamic-depth VDW store.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 31 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 51 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = 1055232 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 33 : i32}
vc4.qpu.vpmvcd_setup {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<add>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 8 : i32,
  raddr_b = 31 : i32,
  add_a = #vc4.qpu_mux<r3>,
  add_b = #vc4.qpu_mux<a>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 48 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r0>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}

// VDW setup: base | (width << 16) | (qpu_id << 7).
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 3 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 31 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 31 : i32, small_imm = 8 : i32, add_a = #vc4.qpu_mux<r1>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.ldi <splat32> {value = -2139078656 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 35 : i32, waddr_mul = 33 : i32}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 30 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<r3>, add_b = #vc4.qpu_mux<r1>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<shl>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 8 : i32, small_imm = 7 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.vpmvcd_setup {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<add>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 31 : i32,
  raddr_b = 30 : i32,
  add_a = #vc4.qpu_mux<r1>,
  add_b = #vc4.qpu_mux<r2>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}

// Store address = output + row_byte_base. Stage row_byte_base through r0
// because the VDW address add reads the output base from ra1.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<or>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 18 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<a>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 33 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, raddr_b = 31 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<r0>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.vpmvcd_addr {
  side = #vc4.vpmvcd_side<write>,
  pm = false,
  cond_add = #vc4.cond<always>,
  cond_mul = #vc4.cond<never>,
  op_add = #vc4.add_opcode<or>,
  op_mul = #vc4.mul_opcode<nop>,
  raddr_a = 30 : i32,
  raddr_b = 31 : i32,
  add_a = #vc4.qpu_mux<r1>,
  add_b = #vc4.qpu_mux<r1>,
  mul_a = #vc4.qpu_mux<r0>,
  mul_b = #vc4.qpu_mux<r1>
}
vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<write>}
vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 51 : i32, waddr_mul = 33 : i32}

// One logical request handles one row for this candidate launch. Retire after
// the row store instead of looping back into the uniform-read region.
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 10 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 10 : i32, raddr_b = 9 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 32 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
}

// end: thread-end epilogue.
vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 28 : i32, raddr_b = 29 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

}
}
