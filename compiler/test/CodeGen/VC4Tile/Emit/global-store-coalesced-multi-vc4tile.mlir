// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses | FileCheck %s

// CHECK-LABEL: vc4.func @global_store_coalesced_multi_vc4tile
// CHECK: vc4.qpu.branch
// CHECK: vc4.qpu.
vc4tile.kernel @global_store_coalesced_multi_vc4tile(%out : i32, %n : i32) attributes {
  public_name = "global_store_coalesced_multi_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %fifteen = arith.constant 15 : i32
  %sixteen = arith.constant 16 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = arith.constant 700 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  %is_one = arith.cmpi eq, %n, %one : i32
  cf.cond_br %is_one, ^store_one, ^not_one
^not_one:
  %is_fifteen = arith.cmpi eq, %n, %fifteen : i32
  cf.cond_br %is_fifteen, ^store_fifteen, ^store_full
^store_full:
  %mask_full = vc4tile.tail_mask %zero, %sixteen : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask_full {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
^store_fifteen:
  %mask_fifteen = vc4tile.tail_mask %zero, %fifteen : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask_fifteen {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
^store_one:
  %mask_one = vc4tile.tail_mask %zero, %one : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask_one {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
