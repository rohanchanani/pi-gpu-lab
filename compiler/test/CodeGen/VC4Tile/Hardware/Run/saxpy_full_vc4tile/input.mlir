// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh saxpy_full_vc4tile generate

vc4tile.kernel @saxpy_full_vc4tile attributes {
  public_name = "saxpy_full_vc4tile",
  arg_attrs = [{name = "buffer", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %loaded = vc4tile.masked_load_global %base, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  %two = arith.constant 2 : i32
  %two_vec = vector.broadcast %two : i32 to vector<16xi32>
  %scaled = arith.muli %loaded, %two_vec : vector<16xi32>
  %result = arith.addi %scaled, %lanes : vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %result, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
