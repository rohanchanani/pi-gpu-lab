// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @multi_exit_early_exit_before_store(
    %n: index {vc4value.arg_name = "n"},
    %limit: index {vc4value.arg_name = "limit"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^loop(%c0 : index)

^loop(%i: index):
  %done = arith.cmpi uge, %i, %n : index
  cf.cond_br %done, ^exit_done, ^check_early(%i : index)

^check_early(%check_i: index):
  %early = arith.cmpi uge, %check_i, %limit : index
  cf.cond_br %early, ^exit_early, ^body(%check_i : index)

^body(%body_i: index):
  %next = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next : index)

^exit_done:
  return

^exit_early:
  return
}

func.func @multi_exit_two_scalar_exits_join(
    %n: index {vc4value.arg_name = "n"},
    %limit: index {vc4value.arg_name = "limit"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %done_code = arith.constant 10 : i32
  %early_code = arith.constant 20 : i32
  cf.br ^loop(%c0 : index)

^loop(%i: index):
  %done = arith.cmpi uge, %i, %n : index
  cf.cond_br %done, ^exit_done, ^check_early(%i : index)

^check_early(%check_i: index):
  %early = arith.cmpi uge, %check_i, %limit : index
  cf.cond_br %early, ^exit_early, ^body(%check_i : index)

^body(%body_i: index):
  %next = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next : index)

^exit_done:
  cf.br ^merge(%done_code : i32)

^exit_early:
  cf.br ^merge(%early_code : i32)

^merge(%code: i32):
  %sink = arith.addi %code, %done_code : i32
  return
}

func.func @multi_exit_vector_accumulator(
    %n: index {vc4value.arg_name = "n"},
    %limit: index {vc4value.arg_name = "limit"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  cf.br ^loop(%c0, %zero : index, vector<16xf32>)

^loop(%i: index, %acc: vector<16xf32>):
  %done = arith.cmpi uge, %i, %n : index
  cf.cond_br %done, ^exit(%acc : vector<16xf32>), ^check_early(%i, %acc : index, vector<16xf32>)

^check_early(%check_i: index, %check_acc: vector<16xf32>):
  %early = arith.cmpi uge, %check_i, %limit : index
  cf.cond_br %early, ^exit(%check_acc : vector<16xf32>), ^body(%check_i, %check_acc : index, vector<16xf32>)

^body(%body_i: index, %body_acc: vector<16xf32>):
  %next_acc = arith.addf %body_acc, %one : vector<16xf32>
  %next_i = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next_i, %next_acc : index, vector<16xf32>)

^exit(%final: vector<16xf32>):
  %sink = arith.addf %final, %zero : vector<16xf32>
  return
}

// VC4K-LABEL: vc4kernel.kernel @multi_exit_early_exit_before_store
// VC4K: cf.cond_br
// VC4K: cf.cond_br
// VC4K: vc4kernel.return
// VC4K-LABEL: vc4kernel.kernel @multi_exit_two_scalar_exits_join
// VC4K: cf.cond_br
// VC4K: ^{{.*}}(%{{.*}}: i32)
// VC4K: vc4kernel.return
// VC4K-LABEL: vc4kernel.kernel @multi_exit_vector_accumulator
// VC4K: cf.cond_br
// VC4K: ^{{.*}}(%{{.*}}: vector<16xf32>)
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

// SCHED-LABEL: vc4.func @multi_exit_early_exit_before_store
// SCHED: vc4.qpu.branch
// SCHED-LABEL: vc4.func @multi_exit_two_scalar_exits_join
// SCHED: vc4.qpu.branch
// SCHED-LABEL: vc4.func @multi_exit_vector_accumulator
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
