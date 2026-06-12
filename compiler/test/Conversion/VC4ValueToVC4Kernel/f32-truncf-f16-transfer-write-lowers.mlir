// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @f32_truncf_f16_transfer_write_lowers(
    %in: memref<?xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.f16_storage_policy = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %loaded = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %bias = arith.constant dense<2.500000e-01> : vector<16xf32>
  %sum = arith.addf %loaded, %bias : vector<16xf32>
  %narrow = arith.truncf %sum : vector<16xf32> to vector<16xf16>
  vector.transfer_write %narrow, %out[%base], %mask {in_bounds = [true]} : vector<16xf16>, memref<?xf16, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @f32_truncf_f16_transfer_write_lowers
// CHECK-SAME: elem_type = "u16"
// CHECK: vc4kernel.fragment_pack
// CHECK-SAME: dest = #vc4kernel.subword_type<f16>
// CHECK-SAME: policy = #vc4kernel.pack_policy<from_f32>
// CHECK: vc4kernel.vpm_write_fragment
// CHECK-SAME: width = #vc4kernel.vpm_width<w16>
// CHECK: vc4kernel.vdw_store_rect_from_vpm
// CHECK-SAME: elem_bytes = 2
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.
