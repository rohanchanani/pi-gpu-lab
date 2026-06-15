// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{scalar global load for attention-apply scale is staged in Phase 16}}
  func.func @attention_apply_scalar_load_staged(
      %scale_ptr: memref<?xf32, #vc4value.global> {vc4value.arg_name = "scale_ptr", vc4value.direction = "in", vc4value.shape_args = ["scale_n"]},
      %scale_n: index {vc4value.arg_name = "scale_n", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.attention_apply_v0 = "scalar_global_load",
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite"} {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %scale = memref.load %scale_ptr[%c0] : memref<?xf32, #vc4value.global>
    return
  }
}
