// RUN: vc4-opt %s --verify-diagnostics

vc4.module @thread_switch_threading_error {
  vc4.func @bad_switch() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{is only legal in functions with threading = threadable}}
    vc4.thread_switch <switch>
    vc4.return
  }
}

vc4.module @semaphore_id_error {
  vc4.func @bad_sema() attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{semaphore 'id' attribute must be in range}}
    vc4.semaphore <acquire> {id = 16 : i32}
    vc4.return
  }
}

vc4.module @async_wait_empty_error {
  vc4.func @bad_wait() attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{requires at least one async token operand}}
    vc4.async.wait
    vc4.return
  }
}

vc4.module @host_interrupt_scheduled_error {
  vc4.func @bad_irq() attributes {threading = 1 : i32, form = 1 : i32} {
    // expected-error@+1 {{is only legal in functions with form = structured}}
    vc4.host_interrupt
    vc4.return
  }
}
