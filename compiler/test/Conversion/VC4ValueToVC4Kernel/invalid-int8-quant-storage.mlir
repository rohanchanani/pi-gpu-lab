// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @int8_quant_storage(
    %in: memref<?xi8, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i8
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xi8, #vc4value.global>, vector<16xi8>
  return
}

// CHECK: int8/int16 quantized storage is staged
// CHECK: READY_FOR_TRITON remains NO
