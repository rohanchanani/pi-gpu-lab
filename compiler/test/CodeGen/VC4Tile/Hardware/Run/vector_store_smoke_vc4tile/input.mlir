// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh vector_store_smoke_vc4tile generate

vc4tile.kernel @vector_store_smoke_vc4tile attributes {
  public_name = "vector_store_smoke_vc4tile",
  launch_abi = {
    public_name = "vector_store_smoke_vc4tile",
    code_symbol = "vector_store_smoke_vc4tile_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [{direction = "by_value", kind = "scalar", name = "out", type = "u32", uniform_index = 0 : i32}],
    builtins = []
  }
} {
  %base = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = arith.constant 100 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
