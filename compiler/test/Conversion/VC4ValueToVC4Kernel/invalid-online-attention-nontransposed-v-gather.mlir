// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_online_attention_nontransposed_v_gather(
    %v: memref<1024xf32, #vc4value.global> {vc4value.arg_name = "v", vc4value.direction = "in"},
    %d: index {vc4value.arg_name = "d", vc4value.scalar_role = "value"},
    %ldv: index {vc4value.arg_name = "ldv", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.i32_mul_policy = "mul24_safe"} {
  %c0 = arith.constant 0 : index
  %pass = arith.constant dense<0.000000e+00> : vector<16xf32>
  %mask = vector.create_mask %ldv : vector<16xi1>
  %lanes = vector.step : vector<16xindex>
  %stride = vector.broadcast %ldv : index to vector<16xindex>
  %d_v = vector.broadcast %d : index to vector<16xindex>
  %offsets0 = arith.muli %lanes, %stride : vector<16xindex>
  %offsets = arith.addi %offsets0, %d_v : vector<16xindex>
  %gathered = vector.gather %v[%c0][%offsets], %mask, %pass {alignment = 4 : i64} : memref<1024xf32, #vc4value.global>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return
}

// CHECK: vector operation 'vector.gather' is not Phase 5 lowerable
