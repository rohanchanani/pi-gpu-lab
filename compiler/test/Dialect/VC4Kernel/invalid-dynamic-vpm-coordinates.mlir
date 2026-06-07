// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @qpu_horizontal_w32_dynamic_x(%arg : i32) attributes {
    public_name = "qpu_horizontal_w32_dynamic_x",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: horizontal VPM QPU access does not encode a word x coordinate
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %arg, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_subword_dynamic_x(%arg : i32) attributes {
    public_name = "qpu_subword_dynamic_x",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: horizontal VPM QPU access does not encode a word x coordinate
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %arg, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @dma_subword_dynamic_x(%ptr : i32, %arg : i32) attributes {
    public_name = "dma_subword_dynamic_x",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "arg", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: dynamic subword VDR DMA x selectors are unproven/deferred in P12
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 dynamic_dst_x %arg {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_subword_missing_selector() attributes {
    public_name = "qpu_subword_missing_selector",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: subword_selector attribute or dynamic subword selector operand is required
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_subword_both_selectors(%arg : i32) attributes {
    public_name = "qpu_subword_both_selectors",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static subword_selector or dynamic subword selector operand, not both
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_subword_selector %arg, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_w32_dynamic_selector(%arg : i32) attributes {
    public_name = "qpu_w32_dynamic_selector",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: 32-bit VPM QPU access must not specify a dynamic subword selector
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_subword_selector %arg, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vertical_subword_dma_deferred(%ptr : i32) attributes {
    public_name = "vertical_subword_dma_deferred",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vertical subword VDR DMA is unproven/deferred in P12
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 4 : i32, global_stride_bytes = 4 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_write_both_x(%arg : i32) attributes { public_name = "qpu_write_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static x or dynamic x operand, not both
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_x %arg, %value, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_write_neither_x() attributes { public_name = "qpu_write_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: static x attribute or dynamic x operand is required
    vc4kernel.vpm_write_fragment %tile, %c0, %value, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_read_both_x(%arg : i32) attributes { public_name = "qpu_read_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static x or dynamic x operand, not both
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %arg, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @qpu_read_neither_x() attributes { public_name = "qpu_read_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: static x attribute or dynamic x operand is required
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdr_load_both_x(%ptr : i32, %arg : i32) attributes { public_name = "vdr_load_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}, {name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static dst_x or dynamic x operand, not both
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 dynamic_dst_x %arg {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdr_load_neither_x(%ptr : i32) attributes { public_name = "vdr_load_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: dst_x attribute or dynamic x operand is required
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdr_rect_both_x(%ptr : i32, %arg : i32) attributes { public_name = "vdr_rect_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}, {name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static dst_x or dynamic x operand, not both
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0 dynamic_dst_x %arg, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdr_rect_neither_x(%ptr : i32) attributes { public_name = "vdr_rect_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: dst_x attribute or dynamic x operand is required
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdw_fragment_both_x(%ptr : i32, %arg : i32) attributes { public_name = "vdw_fragment_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}, {name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static src_x or dynamic x operand, not both
    vc4kernel.vdw_store_vpm_fragment %tile, %c0 dynamic_src_x %arg, %ptr, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdw_fragment_neither_x(%ptr : i32) attributes { public_name = "vdw_fragment_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: src_x attribute or dynamic x operand is required
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %ptr, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdw_rect_both_x(%ptr : i32, %arg : i32) attributes { public_name = "vdw_rect_both_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}, {name = "arg", kind = "scalar", direction = "by_value", type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: specify either static src_x or dynamic x operand, not both
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0 dynamic_src_x %arg, %ptr, %c0, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vdw_rect_neither_x(%ptr : i32) attributes { public_name = "vdw_rect_neither_x", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}], warps_per_block = 1 : i32 } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: src_x attribute or dynamic x operand is required
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
