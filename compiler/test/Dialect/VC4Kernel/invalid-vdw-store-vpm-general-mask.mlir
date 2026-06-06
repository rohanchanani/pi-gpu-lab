// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%ptr : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %cmp = vc4kernel.fragment_cmp %lanes, %lanes {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vdw_store_vpm_fragment does not yet support general-mask predicates
    // Use vc4kernel.vdw_store_fragment for register-fragment general masked stores.
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %ptr, %c0, %cmp {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
