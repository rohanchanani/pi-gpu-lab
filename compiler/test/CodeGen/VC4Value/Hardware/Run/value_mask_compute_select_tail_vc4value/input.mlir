func.func @value_mask_compute_select_tail_vc4value(
    %xf: memref<?xf32, #vc4value.global> {vc4value.arg_name = "xf",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %lo_f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "lo_f",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["n"]},
    %hi_f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "hi_f",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["n"]},
    %out_f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out_f",
                                              vc4value.direction = "out",
                                              vc4value.shape_args = ["n"]},
    %xi: memref<?xi32, #vc4value.global> {vc4value.arg_name = "xi",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %lo_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "lo_i",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["n"]},
    %hi_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "hi_i",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["n"]},
    %out_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out_i",
                                              vc4value.direction = "out",
                                              vc4value.shape_args = ["n"]},
    %threshold_f: f32 {vc4value.arg_name = "threshold_f"},
    %threshold_i: i32 {vc4value.arg_name = "threshold_i"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_f = arith.constant 0.000000e+00 : f32
  %zero_i = arith.constant 0 : i32
  %xfv = vector.transfer_read %xf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %lofv = vector.transfer_read %lo_f[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %hifv = vector.transfer_read %hi_f[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %thfv = vector.broadcast %threshold_f : f32 to vector<16xf32>
  %fm = arith.cmpf olt, %xfv, %thfv : vector<16xf32>
  %fsel = arith.select %fm, %lofv, %hifv : vector<16xi1>, vector<16xf32>
  vector.transfer_write %fsel, %out_f[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  %xiv = vector.transfer_read %xi[%base], %zero_i, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %loiv = vector.transfer_read %lo_i[%base], %zero_i, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %hiiv = vector.transfer_read %hi_i[%base], %zero_i, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %thiv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %im = arith.cmpi sgt, %xiv, %thiv : vector<16xi32>
  %isel = arith.select %im, %hiiv, %loiv : vector<16xi1>, vector<16xi32>
  vector.transfer_write %isel, %out_i[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
