// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

func.func @irreducible_raw_cf(%selector: i32 {vc4value.arg_name = "selector"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %choose_a = arith.cmpi eq, %selector, %c0 : i32
  cf.cond_br %choose_a, ^a(%c0 : i32), ^b(%c1 : i32)

^a(%av: i32):
  %to_b = arith.cmpi eq, %av, %c0 : i32
  cf.cond_br %to_b, ^b(%av : i32), ^exit

^b(%bv: i32):
  %to_a = arith.cmpi eq, %bv, %c1 : i32
  cf.cond_br %to_a, ^a(%bv : i32), ^exit

^exit:
  return
}

// CHECK: SSAVC4 block-argument lowering supports only natural loops with conservative loop-carried data values
