// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @multiblock_k_accumulation_staged_for_phase13
  func.func @multiblock_k_accumulation_staged_for_phase13(
      %a: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["lda"]},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["cols"]},
      %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"},
      %num_kblocks: index {vc4value.arg_name = "num_kblocks", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %row = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    %acc = scf.for %kb = %c0 to %num_kblocks step %c1 iter_args(%running = %zero) -> (f32) {
      %kbase = arith.muli %kb, %c16 : index
      %remaining = arith.subi %cols, %kbase : index
      %mask = vector.create_mask %remaining : vector<16xi1>
      %av = vector.transfer_read %a[%row, %kbase], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
      %xv = vector.transfer_read %x[%kbase], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
      %prod = arith.mulf %av, %xv : vector<16xf32>
      %partial = vector.reduction <add>, %prod : vector<16xf32> into f32
      %next = arith.addf %running, %partial : f32
      scf.yield %next : f32
    }
    memref.store %acc, %y[%row] : memref<?xf32, #vc4value.global>
    return
  }
}
