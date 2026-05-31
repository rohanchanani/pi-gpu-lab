// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%x : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    resource = {
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %xv = vc4kernel.splat %x : i32 -> vector<16xi32>
    %cmp = vc4kernel.fragment_cmp %lanes, %xv {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %cmp : !vc4kernel.pred<16> -> i1
    %all = vc4kernel.pred.all %cmp : !vc4kernel.pred<16> -> i1
    // CHECK: predicate expression is not normalizable
    %sel = vc4kernel.fragment_select %cmp, %lanes, %xv : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}
