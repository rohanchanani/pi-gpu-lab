// RUN: vc4-opt %s --verify-diagnostics

vc4.module @uniform_read_type_error {
  vc4.func @bad_uniform_read() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{result type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    %0 = vc4.uniform.read : i16
    vc4.return
  }
}

vc4.module @uniform_seek_type_error {
  vc4.func @bad_uniform_seek(%offset: vector<16xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{operand must be a scalar signless integer or index}}
    vc4.uniform.seek %offset : vector<16xi32>
    vc4.return
  }
}

vc4.module @mov_type_error {
  vc4.func @bad_mov(%arg0: vector<8xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{result type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    %0 = vc4.mov %arg0 : vector<8xi32>
    vc4.return
  }
}

vc4.module @read_type_error {
  vc4.func @bad_read(%arg0: i16) attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{result type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    %0 = vc4.read %arg0 : i16
    vc4.return
  }
}
