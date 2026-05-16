// RUN: vc4-opt %s --split-input-file --verify-diagnostics

vc4.module @uniform_read_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.uniform.read' is unknown}}
    vc4.uniform.read
  }
}

// -----

vc4.module @tmu_descriptor_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.tmu.descriptor' is unknown}}
    vc4.tmu.descriptor
  }
}

// -----

vc4.module @tmu_request_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.tmu.request' is unknown}}
    vc4.tmu.request
  }
}

// -----

vc4.module @vpm_desc_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.vpm.desc' is unknown}}
    vc4.vpm.desc
  }
}

// -----

vc4.module @dma_desc_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.dma.desc' is unknown}}
    vc4.dma.desc
  }
}

// -----

vc4.module @return_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.return' is unknown}}
    vc4.return
  }
}

// -----

vc4.module @program_end_removed {
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{custom op 'vc4.program_end' is unknown}}
    vc4.program_end
  }
}

// -----

vc4.module @structured_function_form_removed {
  // expected-error@+2 {{expected ::mlir::vc4::FunctionForm to be one of: scheduled}}
  // expected-error@+1 {{failed to parse VC4_FunctionFormAttr parameter 'value'}}
  vc4.func @f() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>}
}
