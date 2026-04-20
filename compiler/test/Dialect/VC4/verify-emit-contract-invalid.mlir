// RUN: vc4-opt --vc4-verify-emit-contract %s --split-input-file --verify-diagnostics

vc4.module @qpu_structured_not_directly_emittable {
  // expected-error@+1 {{is not directly emittable: qasm emission later consumes only domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled> functions}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<threadable>} {
    %0 = vc4.uniform.read : i32
    "vc4.semaphore"() <{id = 2 : i32, mode = #vc4.semaphore_mode<release>}> : () -> ()
    vc4.program_end
    vc4.return
  }
}

// -----

vc4.module @host_scheduled_not_directly_emittable {
  // Generic syntax keeps the function non-external without introducing
  // scheduled-body ops that would trigger unrelated dialect verifiers first.
  // expected-error@+1 {{'vc4.func' op is not directly emittable: launcher generation later consumes only domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured> functions}}
  "vc4.func"() <{domain = #vc4.execution_domain<host>, form = #vc4.function_form<scheduled>, function_type = () -> (), sym_name = "driver_stub", sym_visibility = "private", threading = #vc4.threading_mode<single>}> ({
  ^bb0:
  }) : () -> ()
}
