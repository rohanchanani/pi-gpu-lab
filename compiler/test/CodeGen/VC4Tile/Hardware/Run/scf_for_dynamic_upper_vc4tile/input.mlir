// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck --check-prefix=RAW %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_dynamic_upper_vc4tile clean
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_dynamic_upper_vc4tile generate
// RUN: bash -c 'd="$(bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_dynamic_upper_vc4tile generated-dir)"; FileCheck --check-prefix=CORE %s < "$d/core.vc4tile.mlir"'

// RAW: legalize-vc4tile-core-cfg
// SSAVC4: ssavc4.func @scf_for_dynamic_upper_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4-NOT: scf.
// SSAVC4-NOT: vc4tile.
// CORE-NOT: scf.
// CORE-NOT: index
// CORE-NOT: arith.cmpi eq
// CORE: arith.cmpi ult
// CORE: cf.cond_br
// CORE-NOT: arith.cmpi eq

vc4tile.kernel @scf_for_dynamic_upper_vc4tile attributes {
  public_name = "scf_for_dynamic_upper_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}, {name = "n", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32, %n: i32):
  %lb = arith.constant 0 : index
  %ub = arith.index_cast %n : i32 to index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %total = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (i32) {
    %ii = arith.index_cast %i : index to i32
    %term = arith.addi %ii, %one : i32
    %next = arith.addi %acc, %term : i32
    scf.yield %next : i32
  }

  %lanes = vc4tile.lane_range : vector<16xi32>
  %total_vec = vector.broadcast %total : i32 to vector<16xi32>
  %value = arith.addi %lanes, %total_vec : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
