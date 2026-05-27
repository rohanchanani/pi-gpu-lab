// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_types_layouts_roundtrip
vc4tile.kernel @tile_types_layouts_roundtrip attributes {
  public_name = "tile_types_layouts_roundtrip",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  // CHECK-DAG: #vc4tile.memory_space<global>
  // CHECK-DAG: #vc4tile.memory_space<shared_vpm>
  // CHECK-DAG: #vc4tile.memory_space<register>
  // CHECK-DAG: #vc4tile.layout<row_major>
  // CHECK-DAG: #vc4tile.layout<col_major>
  // CHECK-DAG: #vc4tile.layout<vpm_row>
  // CHECK-DAG: #vc4tile.layout<vpm_col>
  // CHECK-DAG: #vc4tile.layout<transposed_view>
  %global = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 2 : i32,
    element_type = f32, storage_type = f32, expressed_type = f32, accumulator_type = f32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<input>, boundary = #vc4tile.boundary_policy<exact>,
    precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  %shared = vc4tile.tile_descriptor {
    shape = [16, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, boundary = #vc4tile.boundary_policy<tail_predicated>,
    precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  %reg = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<col_major>,
    role = #vc4tile.role<accumulator>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  %vpm_col = vc4tile.tile_descriptor {
    shape = [16, 1], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.layout<vpm_col>,
    role = #vc4tile.role<output>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  %view = vc4tile.tile_descriptor {
    shape = [16, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<transposed_view>,
    role = #vc4tile.role<metadata>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  vc4tile.return
}
