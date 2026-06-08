// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel(%in: memref<16xf32>, %out: memref<16xf32>, %flag: i1) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // CHECK: vc4value.program_id
    %pid = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vc4value.num_programs
    %np = vc4value.num_programs {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.step
    %lane = vector.step : vector<16xindex>
    // CHECK: vector.broadcast
    %splat = vector.broadcast %zero : f32 to vector<16xf32>
    // CHECK: vector.create_mask
    %mask = vector.create_mask %c16 : vector<16xi1>
    // CHECK: vector.transfer_read
    %read = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
    // CHECK: vector.transfer_write
    vector.transfer_write %read, %out[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
    // CHECK: scf.if
    scf.if %flag {
      vector.transfer_write %splat, %out[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
    }
    return
  }
}
