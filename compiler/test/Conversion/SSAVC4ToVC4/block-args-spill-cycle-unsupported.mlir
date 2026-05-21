// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

// CHECK: SSAVC4 block-argument lowering does not yet support cyclic parallel copies involving spilled values
ssavc4.module @block_args_spill_cycle_unsupported {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %a0 = ssavc4.splat %zero : i32 -> vector<16xi32>
    %b0 = ssavc4.splat %one : i32 -> vector<16xi32>
    ssavc4.br ^loop(%a0, %b0 : vector<16xi32>, vector<16xi32>)

  ^loop(%a: vector<16xi32>, %b: vector<16xi32>):
    ssavc4.br ^loop(%b, %a : vector<16xi32>, vector<16xi32>)
  }
}
