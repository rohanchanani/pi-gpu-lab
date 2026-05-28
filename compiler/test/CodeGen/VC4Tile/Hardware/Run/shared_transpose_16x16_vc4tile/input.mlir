// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh shared_transpose_16x16_vc4tile generate
vc4tile.kernel @shared_transpose_16x16_vc4tile attributes {
  public_name = "shared_transpose_16x16_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32,
  arg_attrs = [
    {name = "input", kind = "scalar", direction = "by_value", type = "u32"},
    {name = "output", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
^entry(%input: i32, %output: i32):
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c6 = arith.constant 6 : i32
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %c9 = arith.constant 9 : i32
  %c10 = arith.constant 10 : i32
  %c11 = arith.constant 11 : i32
  %c12 = arith.constant 12 : i32
  %c13 = arith.constant 13 : i32
  %c14 = arith.constant 14 : i32
  %c15 = arith.constant 15 : i32
  %shift = arith.constant 6 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %tile = vc4tile.shared_alloc {
    rows = 16 : i32,
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile
  %row0_bytes = arith.shli %c0, %shift : i32
  %row0_input = arith.addi %input, %row0_bytes : i32
  %row0_values = vc4tile.masked_load_global %row0_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c0, %row0_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row1_bytes = arith.shli %c1, %shift : i32
  %row1_input = arith.addi %input, %row1_bytes : i32
  %row1_values = vc4tile.masked_load_global %row1_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c1, %row1_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row2_bytes = arith.shli %c2, %shift : i32
  %row2_input = arith.addi %input, %row2_bytes : i32
  %row2_values = vc4tile.masked_load_global %row2_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c2, %row2_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row3_bytes = arith.shli %c3, %shift : i32
  %row3_input = arith.addi %input, %row3_bytes : i32
  %row3_values = vc4tile.masked_load_global %row3_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c3, %row3_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row4_bytes = arith.shli %c4, %shift : i32
  %row4_input = arith.addi %input, %row4_bytes : i32
  %row4_values = vc4tile.masked_load_global %row4_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c4, %row4_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row5_bytes = arith.shli %c5, %shift : i32
  %row5_input = arith.addi %input, %row5_bytes : i32
  %row5_values = vc4tile.masked_load_global %row5_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c5, %row5_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row6_bytes = arith.shli %c6, %shift : i32
  %row6_input = arith.addi %input, %row6_bytes : i32
  %row6_values = vc4tile.masked_load_global %row6_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c6, %row6_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row7_bytes = arith.shli %c7, %shift : i32
  %row7_input = arith.addi %input, %row7_bytes : i32
  %row7_values = vc4tile.masked_load_global %row7_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c7, %row7_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row8_bytes = arith.shli %c8, %shift : i32
  %row8_input = arith.addi %input, %row8_bytes : i32
  %row8_values = vc4tile.masked_load_global %row8_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c8, %row8_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row9_bytes = arith.shli %c9, %shift : i32
  %row9_input = arith.addi %input, %row9_bytes : i32
  %row9_values = vc4tile.masked_load_global %row9_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c9, %row9_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row10_bytes = arith.shli %c10, %shift : i32
  %row10_input = arith.addi %input, %row10_bytes : i32
  %row10_values = vc4tile.masked_load_global %row10_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c10, %row10_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row11_bytes = arith.shli %c11, %shift : i32
  %row11_input = arith.addi %input, %row11_bytes : i32
  %row11_values = vc4tile.masked_load_global %row11_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c11, %row11_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row12_bytes = arith.shli %c12, %shift : i32
  %row12_input = arith.addi %input, %row12_bytes : i32
  %row12_values = vc4tile.masked_load_global %row12_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c12, %row12_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row13_bytes = arith.shli %c13, %shift : i32
  %row13_input = arith.addi %input, %row13_bytes : i32
  %row13_values = vc4tile.masked_load_global %row13_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c13, %row13_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row14_bytes = arith.shli %c14, %shift : i32
  %row14_input = arith.addi %input, %row14_bytes : i32
  %row14_values = vc4tile.masked_load_global %row14_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c14, %row14_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %row15_bytes = arith.shli %c15, %shift : i32
  %row15_input = arith.addi %input, %row15_bytes : i32
  %row15_values = vc4tile.masked_load_global %row15_input, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.shared_store %tile, %c15, %row15_values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %col0_values = vc4tile.shared_load %tile, %c0, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row0_output = arith.addi %output, %row0_bytes : i32
  vc4tile.masked_store_global %row0_output, %lanes, %col0_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col1_values = vc4tile.shared_load %tile, %c1, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row1_output = arith.addi %output, %row1_bytes : i32
  vc4tile.masked_store_global %row1_output, %lanes, %col1_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col2_values = vc4tile.shared_load %tile, %c2, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row2_output = arith.addi %output, %row2_bytes : i32
  vc4tile.masked_store_global %row2_output, %lanes, %col2_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col3_values = vc4tile.shared_load %tile, %c3, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row3_output = arith.addi %output, %row3_bytes : i32
  vc4tile.masked_store_global %row3_output, %lanes, %col3_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col4_values = vc4tile.shared_load %tile, %c4, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row4_output = arith.addi %output, %row4_bytes : i32
  vc4tile.masked_store_global %row4_output, %lanes, %col4_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col5_values = vc4tile.shared_load %tile, %c5, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row5_output = arith.addi %output, %row5_bytes : i32
  vc4tile.masked_store_global %row5_output, %lanes, %col5_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col6_values = vc4tile.shared_load %tile, %c6, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row6_output = arith.addi %output, %row6_bytes : i32
  vc4tile.masked_store_global %row6_output, %lanes, %col6_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col7_values = vc4tile.shared_load %tile, %c7, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row7_output = arith.addi %output, %row7_bytes : i32
  vc4tile.masked_store_global %row7_output, %lanes, %col7_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col8_values = vc4tile.shared_load %tile, %c8, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row8_output = arith.addi %output, %row8_bytes : i32
  vc4tile.masked_store_global %row8_output, %lanes, %col8_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col9_values = vc4tile.shared_load %tile, %c9, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row9_output = arith.addi %output, %row9_bytes : i32
  vc4tile.masked_store_global %row9_output, %lanes, %col9_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col10_values = vc4tile.shared_load %tile, %c10, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row10_output = arith.addi %output, %row10_bytes : i32
  vc4tile.masked_store_global %row10_output, %lanes, %col10_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col11_values = vc4tile.shared_load %tile, %c11, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row11_output = arith.addi %output, %row11_bytes : i32
  vc4tile.masked_store_global %row11_output, %lanes, %col11_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col12_values = vc4tile.shared_load %tile, %c12, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row12_output = arith.addi %output, %row12_bytes : i32
  vc4tile.masked_store_global %row12_output, %lanes, %col12_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col13_values = vc4tile.shared_load %tile, %c13, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row13_output = arith.addi %output, %row13_bytes : i32
  vc4tile.masked_store_global %row13_output, %lanes, %col13_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col14_values = vc4tile.shared_load %tile, %c14, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row14_output = arith.addi %output, %row14_bytes : i32
  vc4tile.masked_store_global %row14_output, %lanes, %col14_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %col15_values = vc4tile.shared_load %tile, %c15, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  %row15_output = arith.addi %output, %row15_bytes : i32
  vc4tile.masked_store_global %row15_output, %lanes, %col15_values, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
