// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @f16_store_missing_policy(
      %out: memref<?xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c16 = arith.constant 16 : index
    %base = arith.muli %pid, %c16 : index
    %remaining = arith.subi %n, %base : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %wide = arith.constant dense<0.000000e+00> : vector<16xf32>
    %narrow = arith.truncf %wide : vector<16xf32> to vector<16xf16>
    // expected-error @+1 {{Phase 14 f32 compute to f16 storage requires explicit vc4value.f16_storage_policy = "finite"}}
    vector.transfer_write %narrow, %out[%base], %mask {in_bounds = [true]} : vector<16xf16>, memref<?xf16, #vc4value.global>
    return
  }
}
