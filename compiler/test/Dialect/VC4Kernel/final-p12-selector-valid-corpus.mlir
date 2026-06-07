// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @final_p12_selector_valid_corpus(%in : i32, %out : i32) attributes {
    public_name = "final_p12_selector_valid_corpus",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %x = arith.constant 3 : i32
    %sel8 = arith.constant 2 : i32
    %sel16 = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    // final_qpu_horizontal_w8_selector
    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_subword_selector %sel8, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %qpu_h8 = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_subword_selector %sel8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>

    // final_qpu_horizontal_w16_selector
    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vpm_write_fragment %tile, %c1 dynamic_subword_selector %sel16, %qpu_h8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %qpu_h16 = vc4kernel.vpm_read_fragment %tile, %c1 dynamic_subword_selector %sel16, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>

    // final_qpu_vertical_w8_x_selector
    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_x %x dynamic_subword_selector %sel8, %qpu_h16, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %qpu_v8 = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %x dynamic_subword_selector %sel8, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>

    // final_qpu_vertical_w16_x_selector
    // CHECK: vc4kernel.vpm_write_fragment %{{.*}}, %{{.*}} dynamic_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vpm_write_fragment %tile, %c0 dynamic_x %x dynamic_subword_selector %sel16, %qpu_v8, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %qpu_v16 = vc4kernel.vpm_read_fragment %tile, %c0 dynamic_x %x dynamic_subword_selector %sel16, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>

    // final_vdr_horizontal_w8_x_selector
    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}} {
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x dynamic_subword_selector %sel8 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32

    // final_vdr_horizontal_w16_x_selector
    // CHECK: vc4kernel.vdr_load_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}} {
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c1 dynamic_dst_x %x dynamic_subword_selector %sel16 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 32 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32

    // final_vdr_vertical_w8_x_selector
    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0 dynamic_dst_x %x dynamic_subword_selector %sel8, %c1, %c16, %c16 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32

    // final_vdr_vertical_w16_x_selector
    // CHECK: vc4kernel.vdr_load_rect_to_vpm %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c1 dynamic_dst_x %x dynamic_subword_selector %sel16, %c1, %c16, %c32 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32

    // final_vdw_horizontal_w8_x_selector
    // CHECK: vc4kernel.vdw_store_vpm_fragment %{{.*}}, %{{.*}} dynamic_src_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_vpm_fragment %tile, %c0 dynamic_src_x %x dynamic_subword_selector %sel8, %out, %c0, %full {elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, !vc4kernel.pred<16>

    // final_vdw_horizontal_w16_x_selector
    // CHECK: vc4kernel.vdw_store_vpm_fragment %{{.*}}, %{{.*}} dynamic_src_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_vpm_fragment %tile, %c1 dynamic_src_x %x dynamic_subword_selector %sel16, %out, %c0, %full {elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, !vc4kernel.pred<16>

    // final_vdw_vertical_w8_x_selector
    // CHECK: vc4kernel.vdw_store_rect_from_vpm %{{.*}}, %{{.*}} dynamic_src_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0 dynamic_src_x %x dynamic_subword_selector %sel8, %out, %c0, %c1, %c16, %c16 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    // final_vdw_vertical_w16_x_selector
    // CHECK: vc4kernel.vdw_store_rect_from_vpm %{{.*}}, %{{.*}} dynamic_src_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
    vc4kernel.vdw_store_rect_from_vpm %tile, %c1 dynamic_src_x %x dynamic_subword_selector %sel16, %out, %c0, %c1, %c16, %c32 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    vc4kernel.return
  }
}
