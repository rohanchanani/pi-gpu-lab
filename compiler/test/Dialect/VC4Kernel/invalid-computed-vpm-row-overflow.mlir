// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @user_rows_overflow attributes {
    public_name = "user_rows_overflow",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %tile0 = vc4kernel.vpm_alloc {rows = 64 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: computed VPM row requirement exceeds 64 rows
    %tile1 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @hidden_staging_overflow attributes {
    public_name = "hidden_staging_overflow",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 64 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: computed VPM row requirement exceeds 64 rows
    vc4kernel.vdw_store_fragment %c0, %offs, %v, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
