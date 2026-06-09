// RUN: vc4-opt %s --allow-unregistered-dialect --vc4-verify-value-surface -verify-diagnostics -split-input-file

func.func @vector_condition_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-note @+1 {{prior use here}}
  %mask = arith.constant dense<true> : vector<16xi1>
  // expected-error @+1 {{expects different type than prior uses: 'i1' vs 'vector<16xi1>'}}
  cf.cond_br %mask, ^then, ^else
^then:
  return
^else:
  return
}

// -----

// expected-error @+1 {{memref block arguments are not in the Phase 8 VC4 value control-flow subset}}
func.func @memref_block_argument_reject(
    %buffer: memref<16xf32, #vc4value.global> {vc4value.arg_name = "buffer", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-error @+1 {{memref successor operands are not in the Phase 8 VC4 value control-flow subset}}
  cf.br ^use(%buffer : memref<16xf32, #vc4value.global>)
^use(%arg: memref<16xf32, #vc4value.global>):
  return
}

// -----

func.func @ttir_control_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface: tt}}
  "tt.if"() : () -> ()
  return
}

// -----

func.func @direct_vc4kernel_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-error @+1 {{'vc4kernel.return' op must appear only inside vc4kernel.kernel}}
  "vc4kernel.return"() : () -> ()
}
