// RUN: vc4-opt %s --verify-diagnostics

vc4.module @structured_rejects_scheduled {
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    "vc4.qpu.fake"() : () -> ()
  }
}

vc4.module @scheduled_rejects_structured {
  vc4.func @bad_scheduled() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = structured}}
    %0 = vc4.uniform.read : i32
  }
}

vc4.module @scheduled_rejects_nested_structured {
  vc4.func @bad_nested() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    "vc4.qpu.fake_region"() ({
      // expected-error@+1 {{is only legal in functions with form = structured}}
      %0 = vc4.uniform.read : i32
    }) : () -> ()
  }
}

vc4.module @qpu_domain_rejects_host_ops {
  vc4.func @bad_host_op(%base: i32, %len: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with domain = host}}
    %0 = "vc4.enqueue_qpu"(%base, %len) <{entry = @kernel}> : (i32, i32) -> !vc4.async.token
    vc4.return
  }

  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<single>} {
    vc4.return
  }
}

vc4.module @host_domain_rejects_qpu_ops {
  vc4.func @bad_device_op() attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with domain = qpu}}
    %0 = vc4.uniform.read : i32
    vc4.return
  }
}
