// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_div(%x : i32, %y : i32) attributes {
    public_name = "bad_div",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: integer division and remainder are not supported in vc4kernel scalar arith
    %r = arith.divsi %x, %y : i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_rem(%x : i32, %y : i32) attributes {
    public_name = "bad_rem",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: integer division and remainder are not supported in vc4kernel scalar arith
    %r = arith.remui %x, %y : i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_divui(%x : i32, %y : i32) attributes {
    public_name = "bad_divui",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: integer division and remainder are not supported in vc4kernel scalar arith
    %r = arith.divui %x, %y : i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_remsi(%x : i32, %y : i32) attributes {
    public_name = "bad_remsi",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: integer division and remainder are not supported in vc4kernel scalar arith
    %r = arith.remsi %x, %y : i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vector_arith attributes {
    public_name = "bad_vector_arith",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %lhs = arith.constant dense<1> : vector<16xi32>
    %rhs = arith.constant dense<2> : vector<16xi32>
    // CHECK: arith operations may not operate on or produce vectors
    %r = arith.addi %lhs, %rhs : vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_i64 attributes {
    public_name = "bad_i64",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %lhs = arith.constant 1 : i64
    %rhs = arith.constant 2 : i64
    // CHECK: arith.constant in vc4kernel requires one scalar i1/i32/f32 result
    %r = arith.addi %lhs, %rhs : i64
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_i16 attributes {
    public_name = "bad_i16",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %lhs = arith.constant 1 : i16
    %rhs = arith.constant 2 : i16
    // CHECK: arith.constant in vc4kernel requires one scalar i1/i32/f32 result
    %r = arith.andi %lhs, %rhs : i16
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_i8 attributes {
    public_name = "bad_i8",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %lhs = arith.constant 1 : i8
    %rhs = arith.constant 2 : i8
    // CHECK: arith.constant in vc4kernel requires one scalar i1/i32/f32 result
    %r = arith.ori %lhs, %rhs : i8
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_index attributes {
    public_name = "bad_index",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %lhs = arith.constant 1 : index
    %rhs = arith.constant 2 : index
    // CHECK: arith.constant in vc4kernel requires one scalar i1/i32/f32 result
    %r = arith.addi %lhs, %rhs : index
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f32_add(%x : f32, %y : f32) attributes {
    public_name = "bad_f32_add",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: scalar f32 arithmetic is not supported in vc4kernel scalar arith
    %r = arith.addf %x, %y : f32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f32_sub(%x : f32, %y : f32) attributes {
    public_name = "bad_f32_sub",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: scalar f32 arithmetic is not supported in vc4kernel scalar arith
    %r = arith.subf %x, %y : f32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f32_mul(%x : f32, %y : f32) attributes {
    public_name = "bad_f32_mul",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: scalar f32 arithmetic is not supported in vc4kernel scalar arith
    %r = arith.mulf %x, %y : f32
    vc4kernel.return
  }
}
