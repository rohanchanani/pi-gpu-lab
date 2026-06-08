// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
    scf.execute_region {
      scf.yield
    }
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %flag = arith.constant true
    // expected-error @+1 {{cf operation is not in the Phase 3 VC4 value-surface control subset}}
    cf.assert %flag, "not in value-surface control subset"
    return
  }
}

// -----

builtin.module {
  func.func @kernel(%i: i32 {vc4value.arg_name = "i"}) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{cf operation is not in the Phase 3 VC4 value-surface control subset}}
    cf.switch %i : i32, [
      default: ^bb1
    ]
  ^bb1:
    return
  }
}
