// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @structured_system {
// CHECK: vc4.func @kernel_entry() attributes {form = 0 : i32, kernel, threading = 1 : i32}
// CHECK: vc4.func @driver(%[[BASE:.*]]: i32, %[[LEN:.*]]: i32, %[[SEL:.*]]: i32, %[[CFG:.*]]: i32, %[[SCR:.*]]: i32) attributes {form = 0 : i32, threading = 1 : i32} {
// CHECK: %[[TOK:.*]] = "vc4.enqueue_qpu"(%[[BASE]], %[[LEN]]) <{entry = @kernel_entry}> : (i32, i32) -> !vc4.async.token
// CHECK: vc4.reserve_qpu {mask = 15 : i32}
// CHECK: %[[IDENT:.*]] = "vc4.v3d.query"() <{kind = #vc4.v3d_query_kind<ident>}> : () -> i32
// CHECK: %[[PERF:.*]] = "vc4.v3d.query"(%[[SEL]]) <{kind = #vc4.v3d_query_kind<perf_counter>}> : (i32) -> index
// CHECK: "vc4.v3d.configure"(%[[CFG]]) <{kind = #vc4.v3d_configure_kind<cache_control>}> : (i32) -> ()
// CHECK: "vc4.v3d.configure"(%[[SEL]], %[[SCR]]) <{kind = #vc4.v3d_configure_kind<scratch>}> : (i32, i32) -> ()
// CHECK: "vc4.cf.branch"()[^bb1, ^bb2] <{cond = #vc4.branch_cond<any_z_clear>}> : () -> ()
// CHECK: ^bb1:
// CHECK: "vc4.cf.branch"()[^bb3] <{cond = #vc4.branch_cond<always>}> : () -> ()
// CHECK: ^bb2:
// CHECK: vc4.program_end
// CHECK: ^bb3:
// CHECK: vc4.async.wait %[[TOK]] : !vc4.async.token

vc4.module @structured_system {
  vc4.func @kernel_entry() attributes {kernel, threading = 1 : i32, form = 0 : i32} {
    vc4.program_end
    vc4.return
  }

  vc4.func @driver(%base: i32, %len: i32, %sel: i32, %cfg: i32, %scratch: i32) attributes {threading = 1 : i32, form = 0 : i32} {
    %tok = "vc4.enqueue_qpu"(%base, %len) <{entry = @kernel_entry}> : (i32, i32) -> !vc4.async.token
    "vc4.reserve_qpu"() <{mask = 15 : i32}> : () -> ()
    %ident = "vc4.v3d.query"() <{kind = #vc4.v3d_query_kind<ident>}> : () -> i32
    %perf = "vc4.v3d.query"(%sel) <{kind = #vc4.v3d_query_kind<perf_counter>}> : (i32) -> index
    "vc4.v3d.configure"(%cfg) <{kind = #vc4.v3d_configure_kind<cache_control>}> : (i32) -> ()
    "vc4.v3d.configure"(%sel, %scratch) <{kind = #vc4.v3d_configure_kind<scratch>}> : (i32, i32) -> ()
    "vc4.cf.branch"() [^bb1, ^bb2] <{cond = #vc4.branch_cond<any_z_clear>}> : () -> ()
  ^bb1:
    "vc4.cf.branch"() [^bb3] <{cond = #vc4.branch_cond<always>}> : () -> ()
  ^bb2:
    vc4.program_end
    vc4.return
  ^bb3:
    vc4.async.wait %tok : !vc4.async.token
    vc4.return
  }
}
