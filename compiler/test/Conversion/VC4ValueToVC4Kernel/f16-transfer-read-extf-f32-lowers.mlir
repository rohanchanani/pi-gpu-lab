// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @f16_transfer_read_extf_f32_lowers(
    %in: memref<?xf16, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %loaded = vector.transfer_read %in[%base], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %wide = arith.extf %loaded : vector<16xf16> to vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  %sum = arith.addf %wide, %one : vector<16xf32>
  vector.transfer_write %sum, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @f16_transfer_read_extf_f32_lowers
// CHECK-SAME: elem_type = "u16"
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.vdr_load_rect_to_vpm
// CHECK-SAME: elem_bytes = 2
// CHECK-SAME: memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>
// CHECK: vc4kernel.vpm_read_fragment
// CHECK-SAME: width = #vc4kernel.vpm_width<w16>
// CHECK: vc4kernel.fragment_unpack
// CHECK-SAME: policy = #vc4kernel.unpack_policy<to_f32>
// CHECK-SAME: source = #vc4kernel.subword_type<f16>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<fadd>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.
