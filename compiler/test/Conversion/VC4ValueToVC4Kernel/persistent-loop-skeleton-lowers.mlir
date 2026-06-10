// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @persistent_loop_skeleton_lowers(
    %start_pid: index {vc4value.arg_name = "start_pid"},
    %num_tiles: index {vc4value.arg_name = "num_tiles"},
    %num_sms: index {vc4value.arg_name = "num_sms"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %result = scf.for %tile_id = %start_pid to %num_tiles step %num_sms
      iter_args(%visited = %zero_i32) -> (i32) {
    %next_visited = arith.addi %visited, %one_i32 : i32
    scf.yield %next_visited : i32
  }
  %sink = arith.addi %result, %zero_i32 : i32
  return
}

// VC4K-LABEL: vc4kernel.kernel @persistent_loop_skeleton_lowers
// VC4K: cf.br
// VC4K: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32)
// VC4K: cf.cond_br
// VC4K: arith.addi
// VC4K: vc4kernel.return
// VC4K-NOT: scf.
// VC4K-NOT: func.func
// VC4K-NOT: vector.
// VC4K-NOT: memref.
// VC4K-NOT: vc4value.
// VC4K-NOT: tt.
// VC4K-NOT: ttg.
// VC4K-NOT: gpu.

// SCHED-LABEL: vc4.func @persistent_loop_skeleton_lowers
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
