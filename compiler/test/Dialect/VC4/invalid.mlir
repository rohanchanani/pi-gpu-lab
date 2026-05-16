// RUN: vc4-opt %s --verify-diagnostics --allow-unregistered-dialect

vc4.module @missing_threading {
  // expected-error@+1 {{requires a 'threading' attribute}}
  vc4.func private @f() attributes {form = #vc4.function_form<scheduled>, domain = #vc4.execution_domain<qpu>}
}

vc4.module @missing_form {
  // expected-error@+1 {{requires a 'form' attribute}}
  vc4.func private @f() attributes {threading = #vc4.threading_mode<single>, domain = #vc4.execution_domain<qpu>}
}

vc4.module @missing_domain {
  // expected-error@+1 {{requires a 'domain' attribute}}
  vc4.func private @f() attributes {form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
}

vc4.module @kernel_host_domain_error {
  // expected-error@+1 {{the 'kernel' attribute is only legal with domain = #vc4.execution_domain<qpu>}}
  vc4.func private @f() attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>}
}

vc4.module @structured_form_removed {
  // expected-error@+1 {{structured vc4 form has been removed; use ssavc4 for pre-scheduled SSA IR}}
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = "builtin.unrealized_conversion_cast"() : () -> i32
  }
}

vc4.module @scheduled_host_domain_error {
  // expected-error@+1 {{scheduled VC4 functions require domain = #vc4.execution_domain<qpu>}}
  vc4.func @bad_host_scheduled() attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    %0 = "builtin.unrealized_conversion_cast"() : () -> i32
  }
}

vc4.module @scheduled_non_qpu_error {
  vc4.func @bad_scheduled() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is not legal in scheduled VC4 functions; expected a vc4.qpu.* op}}
    "test.fake"() : () -> ()
  }
}
