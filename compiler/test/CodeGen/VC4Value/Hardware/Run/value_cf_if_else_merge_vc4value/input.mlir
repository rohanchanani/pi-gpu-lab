func.func @value_cf_if_else_merge_vc4value(
    %x: memref<?xi32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %selector: i32 {vc4value.arg_name = "selector"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %cond = arith.cmpi ne, %selector, %zero : i32
  %xv = vector.transfer_read %x[%base], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  cf.cond_br %cond, ^then(%xv : vector<16xi32>), ^else(%xv : vector<16xi32>)

^then(%then_x: vector<16xi32>):
  %plus = arith.constant dense<17> : vector<16xi32>
  %then_value = arith.addi %then_x, %plus : vector<16xi32>
  cf.br ^merge(%then_value : vector<16xi32>)

^else(%else_x: vector<16xi32>):
  %minus = arith.constant dense<23> : vector<16xi32>
  %else_value = arith.subi %else_x, %minus : vector<16xi32>
  cf.br ^merge(%else_value : vector<16xi32>)

^merge(%selected: vector<16xi32>):
  vector.transfer_write %selected, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
