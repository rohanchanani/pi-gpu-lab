// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @cf_vector_condition_invalid()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %mask = arith.constant dense<true> : vector<16xi1>
  cf.cond_br %mask, ^then, ^exit

^then:
  cf.br ^exit

^exit:
  return
}

// CHECK: expects different type than prior uses: 'i1' vs 'vector<16xi1>'
