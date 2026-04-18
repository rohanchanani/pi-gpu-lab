// RUN: vc4-opt %s --verify-diagnostics

vc4.module @branch_successor_error {
  vc4.func @bad_branch() attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{conditional branch requires exactly two successors}}
    "vc4.cf.branch"() [^bb1] <{cond = #vc4.branch_cond<any_z_set>}> : () -> ()
  ^bb1:
    vc4.return
  }
}

vc4.module @enqueue_symbol_error {
  vc4.func @bad_enqueue(%base: i32, %len: i32) attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{referenced 'entry' must resolve to a vc4.func symbol}}
    %0 = "vc4.enqueue_qpu"(%base, %len) <{entry = @missing}> : (i32, i32) -> !vc4.async.token
    vc4.return
  }
}

vc4.module @enqueue_kernel_error {
  vc4.func @not_kernel() attributes {threading = 1 : i32, form = 0 : i32} {
    vc4.return
  }
  vc4.func @bad_kernel_ref(%base: i32, %len: i32) attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{referenced function must be marked with the 'kernel' attribute}}
    %0 = "vc4.enqueue_qpu"(%base, %len) <{entry = @not_kernel}> : (i32, i32) -> !vc4.async.token
    vc4.return
  }
}

vc4.module @reserve_mask_error {
  vc4.func @bad_mask() attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{mask attribute must fit the 12-QPU target range}}
    "vc4.reserve_qpu"() <{mask = 4096 : i32}> : () -> ()
    vc4.return
  }
}

vc4.module @query_selector_error {
  vc4.func @bad_query() attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{selected query kind requires exactly one selector operand}}
    %0 = "vc4.v3d.query"() <{kind = #vc4.v3d_query_kind<perf_counter>}> : () -> i32
    vc4.return
  }
}

vc4.module @configure_payload_error {
  vc4.func @bad_config(%x: i32) attributes {threading = 1 : i32, form = 0 : i32} {
    // expected-error@+1 {{selected configure kind requires exactly 2 payload operands}}
    "vc4.v3d.configure"(%x) <{kind = #vc4.v3d_configure_kind<perf_map>}> : (i32) -> ()
    vc4.return
  }
}
