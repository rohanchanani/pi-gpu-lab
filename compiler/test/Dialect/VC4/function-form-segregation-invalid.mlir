// RUN: vc4-opt %s --verify-diagnostics --allow-unregistered-dialect

vc4.module @structured_form_removed {
  // expected-error@+1 {{structured vc4 form has been removed; use ssavc4 for pre-scheduled SSA IR}}
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = "builtin.unrealized_conversion_cast"() : () -> i32
  }
}

vc4.module @scheduled_rejects_non_qpu {
  vc4.func @bad_scheduled() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is not legal in scheduled VC4 functions; expected a vc4.qpu.* op}}
    "test.fake"() : () -> ()
  }
}

vc4.module @scheduled_requires_qpu_domain {
  // expected-error@+1 {{scheduled VC4 functions require domain = #vc4.execution_domain<qpu>}}
  vc4.func @bad_host_domain() attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    %0 = "builtin.unrealized_conversion_cast"() : () -> i32
  }
}
