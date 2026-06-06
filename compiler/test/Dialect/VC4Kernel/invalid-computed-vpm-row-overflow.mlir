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
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs_shift = vc4kernel.splat %two : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_alu.add %lanes, %offs_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %one : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 64 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: computed VPM row requirement exceeds 64 rows
    vc4kernel.vdw_store_fragment %c0, %offs, %v, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
