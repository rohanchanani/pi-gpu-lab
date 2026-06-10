// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @scf_while_unsupported_body_feature_reject(
    %buffer: memref<16xf32, #vc4value.global> {vc4value.arg_name = "buffer", vc4value.direction = "in"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f32
  %result:2 = scf.while (%i = %c0, %acc = %zero) : (index, f32) -> (index, f32) {
    %keep_going = arith.cmpi ult, %i, %n : index
    scf.condition(%keep_going) %i, %acc : index, f32
  } do {
  ^bb0(%body_i: index, %body_acc: f32):
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %loaded = memref.load %buffer[%body_i] : memref<16xf32, #vc4value.global>
    %next_i = arith.addi %body_i, %c1 : index
    scf.yield %next_i, %loaded : index, f32
  }
  %sink = arith.addf %result#1, %zero : f32
  return
}
