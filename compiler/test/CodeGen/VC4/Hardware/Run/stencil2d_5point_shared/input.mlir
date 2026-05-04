// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the
// stencil2d_5point_shared hardware-run bundle.
//
// Public launcher API:
// stencil2d_5point_shared_launch(state, input, output, width, height,
// tile_origin_x, tile_origin_y,
// center_weight, neighbor_weight)
//
// Semantic reference:
// For a fixed 10 x 14 output tile, compute
// output[y, x] = center_weight * input[y, x] +
// neighbor_weight *
// (input[y - 1, x] + input[y + 1, x] +
// input[y, x - 1] + input[y, x + 1])
// with clamp-to-edge sampling at image boundaries.
//
// Intended cooperative hardware shape:
// 12 resident logical warps, 16 lanes each. VPM rows 0..11 stage a
// 12 x 16 tile including halo rows and halo columns; logical warp rows
// 1..10 and lanes 1..14 produce the 10 x 14 interior tile. Semaphore
// IDs 0..3 and VPM row allocation remain private implementation details.
//
// Physical uniform stream per active QPU:
// [0] input base address
// [1] output base address
// [2] width
// [3] height
// [4] tile_origin_x
// [5] tile_origin_y
// [6] center_weight, carried as one f32/u32 uniform word
// [7] neighbor_weight, carried as one f32/u32 uniform word
// [8] qpu_id builtin suffix word
// [9] num_qpus builtin suffix word
//
// The trusted checked-in qasm for this bundle is intentionally minimal:
// nop; thrend; nop; nop
// This scheduled body preserves that qasm-visible instruction stream while
// carrying the semantic launcher ABI above for future codegen work.

vc4.module @stencil2d_5point_shared {
vc4.func @stencil2d_5point_shared_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "stencil2d_5point_shared_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 10 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "height", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "tile_origin_x", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "tile_origin_y", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32},
{name = "center_weight", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 6 : i32},
{name = "neighbor_weight", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 7 : i32}
],
builtins = [
{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 8 : i32},
{name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 9 : i32}
]
}
} {
// qasm: nop
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

// qasm: thrend
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

// qasm: nop
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

// qasm: nop
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
