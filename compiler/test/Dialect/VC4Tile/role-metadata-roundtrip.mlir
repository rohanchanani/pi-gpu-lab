// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @role_metadata
vc4tile.kernel @role_metadata attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32
} {
  // CHECK-DAG: role = #vc4tile.role<input>
  %input = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<input>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    copy_stage = #vc4tile.copy_stage<prologue>, reuse_hint = #vc4tile.reuse_hint<register>
  } : () -> !vc4tile.tile
  // CHECK-DAG: role = #vc4tile.role<output>
  %output = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<output>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    copy_stage = #vc4tile.copy_stage<epilogue>, reuse_hint = #vc4tile.reuse_hint<producer>
  } : () -> !vc4tile.tile
  // CHECK-DAG: role = #vc4tile.role<accumulator>
  %acc = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<accumulator>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : () -> !vc4tile.tile
  // CHECK-DAG: role = #vc4tile.role<scratch>
  %scratch = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : () -> !vc4tile.tile
  // CHECK-DAG: role = #vc4tile.role<constant>
  %constant = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<constant>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : () -> !vc4tile.tile
  // CHECK-DAG: role = #vc4tile.role<metadata>
  %metadata = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<metadata>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    copy_stage = #vc4tile.copy_stage<none>, reuse_hint = #vc4tile.reuse_hint<none>
  } : () -> !vc4tile.tile
  vc4tile.return
}
