// RUN: vc4-opt %s --allow-unregistered-dialect --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.load' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.load"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.store' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.store"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.tile' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.tile"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.vpm' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.vpm"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.fragment' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.fragment"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.lane_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.lane_id"() : () -> index
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{unregistered operation 'vc4value.warp_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.warp_id"() : () -> index
}
