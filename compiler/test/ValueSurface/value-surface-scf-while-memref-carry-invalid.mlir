// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @scf_while_memref_carry_reject(
    %buffer: memref<16xf32, #vc4value.global> {vc4value.arg_name = "buffer", vc4value.direction = "inout"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %true = arith.constant true
  // expected-error @+1 {{memref block arguments are not in the Phase 8 VC4 value control-flow subset}}
  %result = scf.while (%carried = %buffer) : (memref<16xf32, #vc4value.global>) -> (memref<16xf32, #vc4value.global>) {
    scf.condition(%true) %carried : memref<16xf32, #vc4value.global>
  } do {
  ^bb0(%body_buffer: memref<16xf32, #vc4value.global>):
    scf.yield %body_buffer : memref<16xf32, #vc4value.global>
  }
  return
}
