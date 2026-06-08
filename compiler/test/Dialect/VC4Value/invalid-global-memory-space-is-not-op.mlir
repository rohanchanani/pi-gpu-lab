// RUN: vc4-opt %s -verify-diagnostics

module {
  // expected-error @+1 {{unregistered operation 'vc4value.global' found in dialect ('vc4value') that does not allow unknown operations}}
  "vc4value.global"() : () -> ()
}
