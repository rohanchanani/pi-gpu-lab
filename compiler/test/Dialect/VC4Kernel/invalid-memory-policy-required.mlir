// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @missing_tmu(%ptr : i32) attributes {
    public_name = "missing_tmu",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    %p7_safe0 = arith.constant 0 : i32
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vdw(%ptr : i32) attributes {
    public_name = "missing_vdw",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vpm_write(%ptr : i32) attributes {
    public_name = "missing_vpm_write",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vpm_write_fragment %tile, %c0, %v, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vpm_read(%ptr : i32) attributes {
    public_name = "missing_vpm_read",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    %v = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vdr(%ptr : i32) attributes {
    public_name = "missing_vdr",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vdr_rect(%ptr : i32) attributes {
    public_name = "missing_vdr_rect",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %c4, %c4, %c64 {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vdw_vpm(%ptr : i32) attributes {
    public_name = "missing_vdw_vpm",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %ptr, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_vdw_rect(%ptr : i32) attributes {
    public_name = "missing_vdw_rect",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: requires explicit memory_path and coherency attrs in Surface v2
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %c4, %c4, %c64 {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
