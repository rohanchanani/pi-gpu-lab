// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @cooperative_block_resources
// CHECK: vc4.launch_abi
// CHECK: args =
// CHECK-DAG: name = "out"
// CHECK-DAG: direction = "by_value"
// CHECK-DAG: kind = "scalar"
// CHECK-DAG: uniform_index = 0 : i32
// CHECK: builtins =
// CHECK-DAG: #vc4.builtin_kind<logical_block_id>
// CHECK-DAG: name = "logical_block_id"
// CHECK-DAG: #vc4.builtin_kind<logical_warp_id>
// CHECK-DAG: name = "logical_warp_id"
// CHECK-DAG: #vc4.builtin_kind<warps_per_block>
// CHECK-DAG: name = "warps_per_block"
// CHECK: uniform_words_per_qpu = 4 : i32
// CHECK: vc4.resource
// CHECK-DAG: schedule_mode = "cooperative_block"
// CHECK-DAG: warps_per_block_max = 2 : i32
// CHECK-DAG: uses_barrier = false
// CHECK-DAG: uses_shared_vpm = false
// CHECK: ssavc4.uniform.read 0 : i32
// CHECK: ssavc4.uniform.read 1 : i32
// CHECK: ssavc4.uniform.read 2 : i32
// CHECK: ssavc4.element_number
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @cooperative_block_resources attributes {
  public_name = "cooperative_block_resources",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32,
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %block = vc4tile.block_id : i32
  %warp = vc4tile.warp_id : i32
  %thread = vc4tile.thread_id : vector<16xi32>
  %lanes = vc4tile.lane_range : vector<16xi32>
  %six = arith.constant 6 : i32
  %seven = arith.constant 7 : i32
  %thousand = arith.constant 1000 : i32
  %block_bytes = arith.shli %block, %seven : i32
  %warp_bytes = arith.shli %warp, %six : i32
  %block_scaled = arith.muli %block, %thousand : i32
  %tmp_base = arith.addi %out, %block_bytes : i32
  %base = arith.addi %tmp_base, %warp_bytes : i32
  %block_vec = vector.broadcast %block_scaled : i32 to vector<16xi32>
  %value = arith.addi %block_vec, %thread : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
