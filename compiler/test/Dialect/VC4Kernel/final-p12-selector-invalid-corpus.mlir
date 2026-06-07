// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @vector_dynamic_x(%out : i32) attributes {
    public_name = "vector_dynamic_x",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "out", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: operand #5 must be i32
    vc4kernel.vdw_store_vpm_fragment %tile, %c0 dynamic_src_x %lanes, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x vector<16xi32>, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vector_dynamic_selector() attributes {
    public_name = "vector_dynamic_selector",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: operand #3 must be i32
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_subword_selector %lanes, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
