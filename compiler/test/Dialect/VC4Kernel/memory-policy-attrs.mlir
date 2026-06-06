// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @memory_policy_attrs(%in : i32, %out : i32) attributes {
    public_name = "memory_policy_attrs",
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
    %c64 = arith.constant 64 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %value = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 2 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    // CHECK: vc4kernel.tmu_load_fragment
    // CHECK-SAME: coherency = #vc4kernel.coherency<readonly_tmu>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<tmu_global_read>
    %p7_safe0 = arith.constant 0 : i32
    %loaded = vc4kernel.tmu_load_fragment %in, %offs, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>

    // CHECK: vc4kernel.vpm_write_fragment
    // CHECK-SAME: coherency = #vc4kernel.coherency<vpm_local>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vpm_qpu>
    vc4kernel.vpm_write_fragment %tile, %c0, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vpm_read_fragment
    // CHECK-SAME: coherency = #vc4kernel.coherency<vpm_local>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vpm_qpu>
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>

    // CHECK: vc4kernel.vdr_load_to_vpm
    // CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32

    // CHECK: vc4kernel.vdr_load_rect_to_vpm
    // CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32

    // CHECK: vc4kernel.vdw_store_fragment
    // CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
    // CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vdw_global_store>
    vc4kernel.vdw_store_fragment %out, %offs, %loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vdw_store_vpm_fragment
    // CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
    // CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vdw_global_store>
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>

    // CHECK: vc4kernel.vdw_store_rect_from_vpm
    // CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
    // CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
    // CHECK-SAME: memory_path = #vc4kernel.memory_path<vdw_global_store>
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %out, %c0, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32

    vc4kernel.vdw_store_fragment %out, %offs, %read, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
