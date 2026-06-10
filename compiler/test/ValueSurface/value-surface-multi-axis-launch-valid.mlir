// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @multi_axis_linearized_tail
  func.func @multi_axis_linearized_tail(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                             vc4value.direction = "out",
                                             vc4value.shape_args = ["n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
    // CHECK: vc4value.program_id
    %pid0 = vc4value.program_id {axis = 0 : i32} : index
    %pid1 = vc4value.program_id {axis = 1 : i32} : index
    %pid2 = vc4value.program_id {axis = 2 : i32} : index
    // CHECK: vc4value.num_programs
    %np0 = vc4value.num_programs {axis = 0 : i32} : index
    %np1 = vc4value.num_programs {axis = 1 : i32} : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %mul_z = arith.muli %pid2, %np1 : index
    %yz = arith.addi %mul_z, %pid1 : index
    %mul_yz = arith.muli %yz, %np0 : index
    %block = arith.addi %mul_yz, %pid0 : index
    %base = arith.muli %block, %c16 : index
    // CHECK: vector.step
    %lane = vector.step : vector<16xindex>
    %base_vec = vector.broadcast %base : index to vector<16xindex>
    %idx = arith.addi %base_vec, %lane : vector<16xindex>
    // CHECK: vector.create_mask
    %remaining = arith.subi %n, %base : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %zero = arith.constant dense<0> : vector<16xi32>
    // CHECK: vector.transfer_write
    vector.transfer_write %zero, %out[%base], %mask {in_bounds = [true]} : vector<16xi32>, memref<?xi32, #vc4value.global>
    %sink = arith.addi %base, %c0 : index
    return
  }
}
