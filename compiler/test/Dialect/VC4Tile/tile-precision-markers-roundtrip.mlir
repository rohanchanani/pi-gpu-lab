// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_precision_markers_roundtrip
vc4tile.kernel @tile_precision_markers_roundtrip attributes {
  public_name = "tile_precision_markers_roundtrip",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  // CHECK-DAG: element_type = f32
  // CHECK-DAG: storage_type = f32
  // CHECK-DAG: expressed_type = f32
  // CHECK-DAG: accumulator_type = f32
  // CHECK-DAG: precision = #vc4tile.precision<exact_32>
  // CHECK-DAG: packing = #vc4tile.packing<none>
  %f32 = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 2 : i32,
    element_type = f32, storage_type = f32, expressed_type = f32, accumulator_type = f32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<accumulator>, boundary = #vc4tile.boundary_policy<tail_predicated>,
    precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  // CHECK-DAG: element_type = i32
  // CHECK-DAG: storage_type = i32
  %i32 = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<output>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  vc4tile.return
}
