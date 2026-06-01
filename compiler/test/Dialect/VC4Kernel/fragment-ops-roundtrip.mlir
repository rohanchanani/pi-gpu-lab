// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @frags(%x : i32) attributes {
    public_name = "frags",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: vc4kernel.splat
    %a = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_add
    %add = vc4kernel.fragment_add %a, %a : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_sub
    %sub = vc4kernel.fragment_sub %add, %a : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_mul
    %mul = vc4kernel.fragment_mul %sub, %a : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_shl
    %shl = vc4kernel.fragment_shl %mul, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %shl, %a {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.fragment_select
    %sel = vc4kernel.fragment_select %cmp, %shl, %a : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_rotate
    %rot = vc4kernel.fragment_rotate %sel {amount = 1 : i32} : vector<16xi32> -> vector<16xi32>
    // CHECK: vc4kernel.fragment_reduce
    %red = vc4kernel.fragment_reduce %rot, %cmp {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
