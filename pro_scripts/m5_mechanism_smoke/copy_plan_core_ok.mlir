module {
  vc4tile.kernel @m5_copy_plan_smoke(%out : i32, %in : i32) attributes { public_name = "m5_copy_plan_smoke" } {
    %lanes = vc4tile.lane_range : vector<16xi32>
    %mask = vc4tile.mask_all : vector<16xi1>
    %v = vc4tile.masked_load_global %in, %lanes, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<coalesced>} : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
    vc4tile.masked_store_global %out, %lanes, %v, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<affine_contiguous>} : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
    vc4tile.return
  }
}
