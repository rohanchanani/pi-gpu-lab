// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @dynamic_vpm_coords(%in : i32, %out : i32) attributes {
    public_name = "dynamic_vpm_coords",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %x = arith.constant 3 : i32
    %sel = arith.constant 1 : i32
    %rows = arith.constant 1 : i32
    %cols = arith.constant 16 : i32
    %pitch = arith.constant 64 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_x %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    // CHECK-NOT: x =
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_x %x, %value, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, vector<16xi32>, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vpm_read_fragment %{{.*}}, %{{.*}} dynamic_x %{{.*}}, %{{.*}} {
    %read = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %x, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, !vc4kernel.pred<16> -> vector<16xi32>

    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_subword_selector %sel, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vpm_read_fragment %{{.*}}, %{{.*}} dynamic_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}} {
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    %read_sub = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %x dynamic_subword_selector %sel, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<laned>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>

    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} {
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32

    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} {
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32

    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}} {
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x dynamic_subword_selector %sel {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32

    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32, i32, i32, i32

    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32, i32, i32, i32

    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x dynamic_subword_selector %sel, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32

    // CHECK: vc4kernel.vdw_store_vpm_fragment %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_vpm_fragment %tile, %c0 dynamic_src_x %x, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vdw_store_vpm_fragment %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    vc4kernel.vdw_store_vpm_fragment %tile, %c0 dynamic_src_x %x, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vdw_store_rect_from_vpm %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0 dynamic_src_x %x, %out, %c0, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32

    // CHECK: vc4kernel.vdw_store_rect_from_vpm %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0 dynamic_src_x %x, %out, %c0, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }

  vc4kernel.kernel @static_vpm_coords(%in : i32, %out : i32) attributes {
    public_name = "static_vpm_coords",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %rows = arith.constant 1 : i32
    %cols = arith.constant 16 : i32
    %pitch = arith.constant 64 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    // CHECK: vc4kernel.kernel @static_vpm_coords
    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: x = 0 : i32
    vc4kernel.vpm_write_fragment %tile, %c0, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vpm_read_fragment %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: x = 0 : i32
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>

    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: dst_x = 3 : i32
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32

    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: dst_x = 3 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32

    // CHECK: vc4kernel.vdw_store_vpm_fragment %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: src_x = 3 : i32
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vdw_store_rect_from_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    // CHECK-SAME: src_x = 3 : i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %out, %c0, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
