// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @cooperative_shared_vpm_staging
// CHECK-SAME: logical_warp_id
// CHECK-SAME: vpm_base_row
// CHECK-DAG: user_vpm_rows_per_block = 4 : i32
// CHECK-DAG: compiler_vpm_staging_rows_per_warp = 1 : i32
// CHECK-DAG: total_vpm_rows_per_block = 8 : i32
// CHECK: %[[WARP:.*]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[VPM_BASE:.*]] = ssavc4.uniform.read 2 : i32
// CHECK: ssavc4.vpm.write
// CHECK: %[[BASE_PLUS_USER:.*]] = ssavc4.alu.add %[[VPM_BASE]]
// CHECK: %[[STAGING_ROW:.*]] = ssavc4.alu.add %[[BASE_PLUS_USER]], %[[WARP]]
// CHECK: ssavc4.vdw.store
// CHECK-SAME: %[[STAGING_ROW]]
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @cooperative_shared_vpm_staging(%out : i32) attributes {
    public_name = "cooperative_shared_vpm_staging",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32"}
    ],
    warps_per_block = 4 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %warp = vc4kernel.warp_id : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %warp, %lanes, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %lanes, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
