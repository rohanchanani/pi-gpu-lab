// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the image_boxblur_shared
// hardware-run bundle.
//
// Reference semantics:
// The public launcher computes a fixed 10x14 output tile of a 3x3 grayscale
// box blur over a row-major u32 image, using the low 8 bits of each input
// pixel and clamp-to-edge boundaries:
// out[y,x] = floor((sum_3x3_clamped(input) + 4) / 9)
//
// Public API:
// image_boxblur_shared_prepare(rt, state, max_width, max_height)
// image_boxblur_shared_launch(state, input, output,
// width, height,
// tile_origin_x, tile_origin_y)
//
// Physical/reference-kernel note:
// The trusted qasm currently has no data movement and consists only of a
// safe program-end sequence:
//
// nop; thrend
// nop
// nop
//
// The committed reference launcher implements the exact integer box-blur
// tile semantics on the host side and tracks the one-allocation,
// repeated-launch, 12-QPU, 16-lane, and 12-warp diagnostic fields. This
// input.mlir therefore models the launchable QPU entry as the exact
// scheduled no-op kernel from the trusted qasm while preserving the semantic
// launcher ABI in vc4.launch_abi. The full VPM-tile/barrier QPU algorithm
// described by the test plan is intentionally not invented here because it
// is not present in the trusted qasm source of truth.
//
// Uniform stream layout represented for the semantic launcher ABI:
// [0] input image buffer
// [1] output image buffer
// [2] width
// [3] height
// [4] tile_origin_x
// [5] tile_origin_y

vc4.module @image_boxblur_shared {
vc4.func @image_boxblur_shared_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "image_boxblur_shared_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 6 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
{name = "width", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "height", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
{name = "tile_origin_x", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
{name = "tile_origin_y", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
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
