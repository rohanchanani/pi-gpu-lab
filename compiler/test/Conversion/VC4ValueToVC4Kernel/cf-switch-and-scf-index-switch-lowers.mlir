// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @scf_index_switch_lowers(
    %selector: index {vc4value.arg_name = "selector"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %result = scf.index_switch %selector -> i32
  case 0 {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  case 1 {
    %one = arith.constant 1 : i32
    scf.yield %one : i32
  }
  default {
    %other = arith.constant 2 : i32
    scf.yield %other : i32
  }
  %sink = arith.addi %result, %result : i32
  return
}

func.func @cf_switch_lowers(%selector: i32 {vc4value.arg_name = "selector"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  cf.switch %selector : i32, [
    default: ^default,
    0: ^case0,
    1: ^case1
  ]

^case0:
  %zero = arith.constant 0 : i32
  cf.br ^exit(%zero : i32)

^case1:
  %one = arith.constant 1 : i32
  cf.br ^exit(%one : i32)

^default:
  %other = arith.constant 2 : i32
  cf.br ^exit(%other : i32)

^exit(%result: i32):
  %sink = arith.addi %result, %result : i32
  return
}

// VC4K-LABEL: vc4kernel.kernel @scf_index_switch_lowers
// VC4K: cf.cond_br
// VC4K: cf.br
// VC4K: ^{{.*}}(%{{.*}}: i32)
// VC4K: vc4kernel.return
// VC4K-LABEL: vc4kernel.kernel @cf_switch_lowers
// VC4K: cf.cond_br
// VC4K: cf.br
// VC4K: ^{{.*}}(%{{.*}}: i32)
// VC4K: vc4kernel.return
// VC4K-NOT: cf.switch
// VC4K-NOT: scf.
// VC4K-NOT: func.func
// VC4K-NOT: vector.
// VC4K-NOT: memref.
// VC4K-NOT: vc4value.
// VC4K-NOT: tt.
// VC4K-NOT: ttg.
// VC4K-NOT: gpu.

// SCHED-LABEL: vc4.func @scf_index_switch_lowers
// SCHED: vc4.qpu.branch
// SCHED-LABEL: vc4.func @cf_switch_lowers
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
