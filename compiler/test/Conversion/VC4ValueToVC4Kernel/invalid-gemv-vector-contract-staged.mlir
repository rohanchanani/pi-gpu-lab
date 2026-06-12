// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

#vec = affine_map<(i) -> (i)>
#scalar = affine_map<(i) -> ()>

func.func @invalid_gemv_vector_contract_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %a = arith.constant dense<1.000000e+00> : vector<16xf32>
  %b = arith.constant dense<2.000000e+00> : vector<16xf32>
  %acc = arith.constant 0.000000e+00 : f32
  %contract = vector.contract {indexing_maps = [#vec, #vec, #scalar], iterator_types = ["reduction"]} %a, %b, %acc : vector<16xf32>, vector<16xf32> into f32
  return
}

// CHECK: vector.contract is staged for Phase 15/contract
// CHECK: READY_FOR_TRITON remains NO
