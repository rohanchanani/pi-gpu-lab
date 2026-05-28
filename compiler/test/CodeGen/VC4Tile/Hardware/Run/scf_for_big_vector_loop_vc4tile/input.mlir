// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck --check-prefix=RAW %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_big_vector_loop_vc4tile clean
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_big_vector_loop_vc4tile generate
// RUN: bash -c 'd="$(bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_big_vector_loop_vc4tile generated-dir)"; FileCheck --check-prefix=CORE %s < "$d/core.vc4tile.mlir"'

// RAW: legalize-vc4tile-core-cfg
// SSAVC4: ssavc4.func @scf_for_big_vector_loop_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4-NOT: scf.
// SSAVC4-NOT: vc4tile.
// CORE-NOT: scf.
// CORE-NOT: index
// CORE-NOT: arith.cmpi eq
// CORE: arith.cmpi ult
// CORE: cf.cond_br
// CORE-NOT: arith.cmpi eq

vc4tile.kernel @scf_for_big_vector_loop_vc4tile attributes {
  public_name = "scf_for_big_vector_loop_vc4tile",
  arg_attrs = [
    {name = "out", kind = "scalar", direction = "by_value", type = "u32"},
    {name = "loop_n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
^entry(%out: i32, %loop_n: i32):
  %pid = vc4tile.program_id : i32
  %four = arith.constant 4 : i32
  %six = arith.constant 6 : i32
  %pid_lanes = arith.shli %pid, %four : i32
  %pid_bytes = arith.shli %pid, %six : i32
  %base = arith.addi %out, %pid_bytes : i32

  %lanes = vc4tile.lane_range : vector<16xi32>
  %pid_lanes_vec = vector.broadcast %pid_lanes : i32 to vector<16xi32>
  %initial = arith.addi %pid_lanes_vec, %lanes : vector<16xi32>

  %lb = arith.constant 0 : index
  %ub = arith.index_cast %loop_n : i32 to index
  %step = arith.constant 1 : index
  %value = scf.for %i = %lb to %ub step %step iter_args(%acc = %initial) -> (vector<16xi32>) {
    %ii = arith.index_cast %i : index to i32
    %s = vector.broadcast %ii : i32 to vector<16xi32>
    %once = arith.addi %acc, %s : vector<16xi32>
    %twice = arith.addi %once, %s : vector<16xi32>
    scf.yield %twice : vector<16xi32>
  }

  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
