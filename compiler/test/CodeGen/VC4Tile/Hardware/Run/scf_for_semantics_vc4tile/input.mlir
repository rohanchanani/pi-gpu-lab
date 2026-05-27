// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck --check-prefix=RAW %s
// RUN: vc4-opt --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_semantics_vc4tile clean
// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_semantics_vc4tile generate
// RUN: bash -c 'd="$(bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_for_semantics_vc4tile generated-dir)"; FileCheck --check-prefix=CORE %s < "$d/core.vc4tile.mlir"'

// RAW: legalize-vc4tile-core-cfg
// SSAVC4: ssavc4.func @scf_for_semantics_vc4tile
// SSAVC4: ssavc4.cond_br
// SSAVC4-NOT: scf.
// SSAVC4-NOT: vc4tile.
// CORE-NOT: scf.
// CORE-NOT: index
// CORE-NOT: arith.cmpi eq
// CORE: arith.cmpi ult
// CORE: cf.cond_br
// CORE-NOT: arith.cmpi eq

vc4tile.kernel @scf_for_semantics_vc4tile attributes {
  public_name = "scf_for_semantics_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %zero = arith.constant 0 : i32
  %lb_exact = arith.constant 0 : index
  %ub_exact = arith.constant 4 : index
  %step_exact = arith.constant 1 : index
  %exact = scf.for %i_exact = %lb_exact to %ub_exact step %step_exact iter_args(%acc_exact = %zero) -> (i32) {
    %ii_exact = arith.index_cast %i_exact : index to i32
    %next_exact = arith.addi %acc_exact, %ii_exact : i32
    scf.yield %next_exact : i32
  }
  %zlb = arith.constant 4 : index
  %zub = arith.constant 0 : index
  %zstep = arith.constant 1 : index
  %seven = arith.constant 7 : i32
  %zero_trip = scf.for %zi = %zlb to %zub step %zstep iter_args(%zacc = %seven) -> (i32) {
    %zii = arith.index_cast %zi : index to i32
    %znext = arith.addi %zacc, %zii : i32
    scf.yield %znext : i32
  }
  %nlb = arith.constant 0 : index
  %nub = arith.constant 5 : index
  %nstep = arith.constant 2 : index
  %ten = arith.constant 10 : i32
  %one = arith.constant 1 : i32
  %nondiv = scf.for %ni = %nlb to %nub step %nstep iter_args(%nacc = %zero) -> (i32) {
    %nii = arith.index_cast %ni : index to i32
    %scaled = arith.muli %nii, %ten : i32
    %term = arith.addi %scaled, %one : i32
    %nnext = arith.addi %nacc, %term : i32
    scf.yield %nnext : i32
  }
  %tmp = arith.addi %exact, %zero_trip : i32
  %total = arith.addi %tmp, %nondiv : i32

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
