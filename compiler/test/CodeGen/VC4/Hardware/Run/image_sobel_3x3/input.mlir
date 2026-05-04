// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage VC4 scheduled-sink input for the image_sobel_3x3
// hardware-run bundle.
//
// Reference semantics:
// The public launcher computes a grayscale Sobel edge detector over a
// row-major u32 image, using the low 8 bits of each input pixel and
// clamp-to-edge boundaries:
// gx = -p00 + p02 - 2p10 + 2p12 - p20 + p22
// gy = -p00 - 2p01 - p02 + p20 + 2p21 + p22
// out = min(255, abs(gx) + abs(gy))
//
// Public API:
// image_sobel_3x3_prepare(rt, state, max_width, max_height)
// image_sobel_3x3_launch(state, input, output, width, height)
// image_sobel_3x3_shutdown(state)
//
// Trusted hardware source of truth:
// The reference qasm for this committed bundle is a launchable no-op program:
// nop; thrend
// nop
// nop
//
// The reference launcher implements the exact integer Sobel semantics on the
// host side and tracks the one-allocation/repeated-launch diagnostics. This
// final-stage input.mlir therefore models the trusted launchable QPU entry
// exactly as scheduled VC4 sink operations while preserving the public
// semantic launcher ABI in vc4.launch_abi. The direct-TMU raster Sobel QPU
// algorithm described by the test plan is not invented here because it is not
// present in the trusted qasm.
//
// Semantic uniform stream represented for the launcher ABI:
// [0] input image buffer
// [1] output image buffer
// [2] width
// [3] height
//
// The public API does not expose total_pixels, qpu_id, num_qpus, raw uniform
// arrays, scratch layout, or QPU scheduling internals in the committed
// reference launcher. Those values remain private runtime policy.

vc4.module @image_sobel_3x3 {
vc4.func @image_sobel_3x3_kernel() attributes {
domain = #vc4.execution_domain<qpu>,
form = #vc4.function_form<scheduled>,
kernel,
threading = #vc4.threading_mode<single>,
"vc4.launch_abi" = {
public_name = "image_sobel_3x3_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 4 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
{name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
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
