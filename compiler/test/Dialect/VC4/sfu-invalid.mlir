// RUN: vc4-opt %s --verify-diagnostics

vc4.module @sfu_issue_type_error {
  vc4.func @bad_issue(%arg0: i16) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    vc4.sfu.issue <recip> %arg0 : i16 // expected-error {{input type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    vc4.return
  }
}

vc4.module @sfu_issue_shape_error {
  vc4.func @bad_issue_shape(%arg0: vector<8xf32>) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    vc4.sfu.issue <log> %arg0 : vector<8xf32> // expected-error {{input type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    vc4.return
  }
}

vc4.module @sfu_read_type_error {
  vc4.func @bad_read() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = vc4.sfu.read : i16 // expected-error {{result type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    vc4.return
  }
}

vc4.module @sfu_read_shape_error {
  vc4.func @bad_read_shape() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = vc4.sfu.read : vector<8xi32> // expected-error {{result type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    vc4.return
  }
}
