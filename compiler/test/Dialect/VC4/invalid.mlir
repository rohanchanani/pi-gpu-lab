// RUN: not vc4-opt %s --verify-diagnostics --allow-unregistered-dialect

vc4.module @missing_threading {
  // expected-error@+1 {{requires a 'threading' attribute}}
  vc4.func private @f() attributes {form = 0 : i32}
}

vc4.module @missing_form {
  // expected-error@+1 {{requires a 'form' attribute}}
  vc4.func private @f() attributes {threading = 0 : i32}
}

vc4.module @builtin_type_error {
  vc4.func @bad_builtin() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{result type must be i32 or vector<16xi32>}}
    %0 = vc4.builtin elem_num : vector<8xi32>
    vc4.return
  }
}

vc4.module @scheduled_return_error {
  vc4.func @bad_return() attributes {threading = 0 : i32, form = 1 : i32} {
    // expected-error@+1 {{is only legal in functions with form = structured}}
    vc4.return
  }
}

vc4.module @return_type_error {
  vc4.func @bad_result() -> i32 attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.builtin elem_num : vector<16xi32>
    // expected-error@+1 {{type of return operand #0 (vector<16xi32>) must match the enclosing function result type (i32)}}
    vc4.return %0 : vector<16xi32>
  }
}

vc4.module @structured_qpu_error {
  vc4.func @bad_structured() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    "vc4.qpu.fake"() : () -> ()
  }
}

vc4.module @scheduled_non_qpu_error {
  vc4.func @bad_scheduled() attributes {threading = 0 : i32, form = 1 : i32} {
    // expected-error@+1 {{is not a legal operation in functions with form = scheduled}}
    "test.fake"() : () -> ()
  }
}
