// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_attention_apply_scalar_global_load(
    %scale_ptr: memref<?xf32, #vc4value.global> {vc4value.arg_name = "scale_ptr", vc4value.direction = "in", vc4value.shape_args = ["scale_n"]},
    %scale_n: index {vc4value.arg_name = "scale_n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.attention_apply_v0 = "scalar_global_load",
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite"} {
  %c0 = arith.constant 0 : index
  %scale = memref.load %scale_ptr[%c0] : memref<?xf32, #vc4value.global>
  return
}

// CHECK: direct memref side-effect operation is not legal in the VC4 value surface
// CHECK: scalar global load for attention-apply scale is staged in Phase 16
