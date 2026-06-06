// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_same_type_bitcast(%x : i32) attributes {
    public_name = "bad_same_type_bitcast",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arith.bitcast requires scalar i32/f32 reinterpretation in vc4kernel
    %bad = arith.bitcast %x : i32 to i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vector_bitcast attributes {
    public_name = "bad_vector_bitcast",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    // CHECK: arith operations may not operate on or produce vectors
    %bad = arith.bitcast %v : vector<16xi32> to vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_extui(%x : i32) attributes {
    public_name = "bad_extui",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arith.extui in vc4kernel supports only i1 to i32 in P5
    %bad = arith.extui %x : i32 to i64
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_trunci(%x : i32) attributes {
    public_name = "bad_trunci",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arith.trunci in vc4kernel supports only i32 to i1 low-bit trunc in P5
    %bad = arith.trunci %x : i32 to i16
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_sitofp(%x : i32) attributes {
    public_name = "bad_sitofp",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arith.sitofp requires exact scalar numeric cast support not available in P5
    %bad = arith.sitofp %x : i32 to f32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_fptosi(%x : f32) attributes {
    public_name = "bad_fptosi",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    // CHECK: arith.fptosi requires exact scalar numeric cast support not available in P5
    %bad = arith.fptosi %x : f32 to i32
    vc4kernel.return
  }
}
