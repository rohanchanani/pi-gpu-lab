// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_tmu_path(%ptr : i32) attributes {
    public_name = "bad_tmu_path",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: memory_path attr does not match operation
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full {memory_path = #vc4kernel.memory_path<vpm_qpu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vpm_coherency(%ptr : i32) attributes {
    public_name = "bad_vpm_coherency",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: coherency attr does not match operation
    vc4kernel.vpm_write_fragment %tile, %c0, %v, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vdr_path(%ptr : i32) attributes {
    public_name = "bad_vdr_path",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: memory_path attr does not match operation
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<tmu_global_read>} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vdw_coherency(%ptr : i32) attributes {
    public_name = "bad_vdw_coherency",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    // CHECK: coherency attr does not match operation
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %full {coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_internal_path(%ptr : i32) attributes {
    public_name = "bad_internal_path",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: compiler spill memory path is internal to the lower half
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full {memory_path = #vc4kernel.memory_path<compiler_spill_vdw_vdr>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_internal_coherency(%ptr : i32) attributes {
    public_name = "bad_internal_coherency",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: compiler spill memory path is internal to the lower half
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full {coherency = #vc4kernel.coherency<compiler_spill_coherent>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
