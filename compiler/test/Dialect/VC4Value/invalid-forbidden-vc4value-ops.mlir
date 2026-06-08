// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{unregistered operation 'vc4value.load' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.load"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.store' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.store"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.tile' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.tile"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.vpm' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.vpm"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.tmu' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.tmu"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.vdr' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.vdr"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.vdw' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.vdw"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.fragment' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.fragment"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.lane_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.lane_id"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.warp_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.warp_id"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.thread_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.thread_id"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.block_id' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.block_id"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.barrier' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.barrier"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'vc4value.semaphore' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.semaphore"() : () -> ()
}
