// RUN: vc4-opt %s --verify-diagnostics

vc4.module @structured_rejects_scheduled {
  vc4.func @bad_structured() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    "vc4.qpu.fake"() : () -> ()
  }
}

vc4.module @scheduled_rejects_structured {
  vc4.func @bad_scheduled() attributes {threading = 0 : i32, form = 1 : i32} {
    // expected-error@+1 {{is only legal in functions with form = structured}}
    %0 = vc4.uniform.read : i32
  }
}

vc4.module @scheduled_rejects_nested_structured {
  vc4.func @bad_nested() attributes {threading = 0 : i32, form = 1 : i32} {
    "vc4.qpu.fake_region"() ({
      // expected-error@+1 {{is only legal in functions with form = structured}}
      %0 = vc4.uniform.read : i32
    }) : () -> ()
  }
}
