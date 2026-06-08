// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

// expected-error @+1 {{expected at least one func.func marked with vc4value.kernel}}
builtin.module {
  func.func @helper() {
    return
  }
}
