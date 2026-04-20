// RUN: vc4-opt %s --verify-diagnostics --allow-unregistered-dialect

vc4.module @missing_threading {
  // expected-error@+1 {{requires a 'threading' attribute}}
  vc4.func private @f() attributes {form = #vc4.function_form<structured>, domain = #vc4.execution_domain<qpu>}
}

vc4.module @missing_form {
  // expected-error@+1 {{requires a 'form' attribute}}
  vc4.func private @f() attributes {threading = #vc4.threading_mode<single>, domain = #vc4.execution_domain<qpu>}
}

vc4.module @missing_domain {
  // expected-error@+1 {{requires a 'domain' attribute}}
  vc4.func private @f() attributes {form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>}
}

vc4.module @kernel_host_domain_error {
  // expected-error@+1 {{the 'kernel' attribute is only legal with domain = #vc4.execution_domain<qpu>}}
  vc4.func private @f() attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<single>}
}

vc4.module @builtin_type_error {
  vc4.func @bad_builtin() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{result type must be i32 or vector<16xi32>}}
    %0 = vc4.builtin elem_num : vector<8xi32>
    vc4.return
  }
}

vc4.module @scheduled_return_error {
  vc4.func @bad_return() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = structured}}
    vc4.return
  }
}

vc4.module @return_type_error {
  vc4.func @bad_result() -> i32 attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = vc4.builtin elem_num : vector<16xi32>
    // expected-error@+1 {{type of return operand #0}}
    vc4.return %0 : vector<16xi32>
  }
}

vc4.module @structured_qpu_error {
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    "vc4.qpu.fake"() : () -> ()
  }
}

vc4.module @scheduled_non_qpu_error {
  vc4.func @bad_scheduled() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is not a legal operation in functions with form = scheduled}}
    "test.fake"() : () -> ()
  }
}
