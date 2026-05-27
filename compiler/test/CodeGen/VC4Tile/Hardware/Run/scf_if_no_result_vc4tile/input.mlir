// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck --check-prefix=RAW %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_if_no_result_vc4tile clean
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_if_no_result_vc4tile generate
// RUN: bash -c 'd="$(bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_if_no_result_vc4tile generated-dir)"; FileCheck --check-prefix=CORE %s < "$d/core.vc4tile.mlir"'

// RAW: legalize-vc4tile-core-cfg
// SSAVC4: ssavc4.func @scf_if_no_result_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4-NOT: scf.
// SSAVC4-NOT: vc4tile.
// CORE-NOT: scf.
// CORE-NOT: index
// CORE-NOT: arith.cmpi eq
// CORE: arith.cmpi ne
// CORE: cf.cond_br
// CORE-NOT: arith.cmpi eq

vc4tile.kernel @scf_if_no_result_vc4tile attributes {
  public_name = "scf_if_no_result_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}, {name = "flag", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32, %flag: i32):
  %zero = arith.constant 0 : i32
  %cond = arith.cmpi ne, %flag, %zero : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  scf.if %cond {
    %then_total = arith.constant 28 : i32
    %then_vec = vector.broadcast %then_total : i32 to vector<16xi32>
    %then_value = arith.addi %lanes, %then_vec : vector<16xi32>
    vc4tile.masked_store_global %out, %lanes, %then_value, %mask {
      elem_bytes = 4 : i32,
      offset_unit = #vc4tile.offset_unit<element>,
      memory_space = #vc4tile.memory_space<global>,
      access = #vc4tile.memory_access<coalesced>
    } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  } else {
    %else_total = arith.constant 3 : i32
    %else_vec = vector.broadcast %else_total : i32 to vector<16xi32>
    %else_value = arith.addi %lanes, %else_vec : vector<16xi32>
    vc4tile.masked_store_global %out, %lanes, %else_value, %mask {
      elem_bytes = 4 : i32,
      offset_unit = #vc4tile.offset_unit<element>,
      memory_space = #vc4tile.memory_space<global>,
      access = #vc4tile.memory_access<coalesced>
    } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  }
  vc4tile.return
}
