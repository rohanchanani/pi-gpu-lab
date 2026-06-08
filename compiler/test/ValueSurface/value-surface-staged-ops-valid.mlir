// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

#mat = affine_map<(m, n, k) -> (m, k)>
#vec = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel(
      %mem: memref<16xf32>,
      %lhs: vector<16xf32>,
      %rhs: vector<16xf32>,
      %a: vector<1x4xf32>,
      %b: vector<4x16xf32>,
      %c: vector<1x16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    %idx = vector.step : vector<16xindex>
    %mask = vector.create_mask %c16 : vector<16xi1>
    %pass = vector.broadcast %zero : f32 to vector<16xf32>
    // CHECK: vector.gather
    %g = vector.gather %mem[%c0][%idx], %mask, %pass {alignment = 4 : i64} : memref<16xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
    // CHECK: vector.reduction
    %r = vector.reduction <add>, %lhs : vector<16xf32> into f32
    // CHECK: math.exp
    %e = math.exp %r : f32
    // CHECK: vector.shuffle
    %sh = vector.shuffle %lhs, %rhs [0, 16, 1, 17] : vector<16xf32>, vector<16xf32>
    // CHECK: vector.contract
    %contract = vector.contract {indexing_maps = [#mat, #vec, #out], iterator_types = ["parallel", "parallel", "reduction"]} %a, %b, %c : vector<1x4xf32>, vector<4x16xf32> into vector<1x16xf32>
    return
  }
}
