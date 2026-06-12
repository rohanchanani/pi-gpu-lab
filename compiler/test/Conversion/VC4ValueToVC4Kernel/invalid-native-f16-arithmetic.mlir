// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @native_f16_arithmetic(
    %a: memref<?xf16, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %b: memref<?xf16, #vc4value.global> {vc4value.arg_name = "b", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.f16_storage_policy = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %av = vector.transfer_read %a[%base], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %bv = vector.transfer_read %b[%base], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %sum = arith.addf %av, %bv : vector<16xf16>
  vector.transfer_write %sum, %out[%base], %mask {in_bounds = [true]} : vector<16xf16>, memref<?xf16, #vc4value.global>
  return
}

// CHECK: native f16 arithmetic is staged
// CHECK: READY_FOR_TRITON remains NO
