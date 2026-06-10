// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @tl_range_style_loop_skeleton_lowers(
    %start: index {vc4value.arg_name = "start"},
    %end: index {vc4value.arg_name = "end"},
    %step: index {vc4value.arg_name = "step"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant dense<0> : vector<16xi32>
  %one = arith.constant dense<1> : vector<16xi32>
  %result = scf.for %tile = %start to %end step %step iter_args(%acc = %zero) -> (vector<16xi32>) {
    %next = arith.addi %acc, %one : vector<16xi32>
    scf.yield %next : vector<16xi32>
  }
  %sink = arith.addi %result, %zero : vector<16xi32>
  return
}

// VC4K-LABEL: vc4kernel.kernel @tl_range_style_loop_skeleton_lowers
// VC4K: cf.br
// VC4K: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xi32>)
// VC4K: cf.cond_br
// VC4K: vc4kernel.fragment_alu.add
// VC4K: vc4kernel.return
// VC4K-NOT: scf.
// VC4K-NOT: func.func
// VC4K-NOT: vector.
// VC4K-NOT: memref.
// VC4K-NOT: vc4value.
// VC4K-NOT: tt.
// VC4K-NOT: ttg.
// VC4K-NOT: gpu.

// SCHED-LABEL: vc4.func @tl_range_style_loop_skeleton_lowers
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
