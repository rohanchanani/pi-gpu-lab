// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh warp_reduce_sum_vc4tile generate

vc4tile.kernel @warp_reduce_sum_vc4tile attributes {
  public_name = "warp_reduce_sum_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %sum = vc4tile.reduce %lanes, %mask {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %sum, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
