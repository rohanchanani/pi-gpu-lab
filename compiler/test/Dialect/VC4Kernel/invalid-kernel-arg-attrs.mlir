// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @length(%x : i32) attributes {
    public_name = "length",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: arg_attrs length must match formal argument count
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @duplicate(%x : i32, %y : i32) attributes {
    public_name = "duplicate",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "dup", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "dup", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: duplicate arg_attrs name 'dup'
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @uniform(%x : i32) attributes {
    public_name = "uniform",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32", uniform_index = 0 : i32}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arg_attrs must not contain uniform_index
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @buffer_type(%ptr : f32) attributes {
    public_name = "buffer_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: buffer args must be i32 raw device pointers
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @scalar_mismatch(%x : f32) attributes {
    public_name = "scalar_mismatch",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: scalar i32/u32 arg must have i32 formal type
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_type(%x : i32) attributes {
    public_name = "missing_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: scalar arg_attrs require type and must not use elem_type
    vc4kernel.return
  }
}
