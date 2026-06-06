// RUN: not vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @vdw_explicit_sparse_reject(%out : i32, %threshold : i32) attributes {
    public_name = "vdw_explicit_sparse_reject",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>
    %mask = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: sparse VDW store masks are not supported in P8
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %mask {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
