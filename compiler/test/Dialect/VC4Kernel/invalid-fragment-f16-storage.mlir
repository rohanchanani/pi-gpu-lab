// RUN: vc4-opt %s --verify-vc4kernel --split-input-file -verify-diagnostics

module {
  vc4kernel.kernel @bad_f16_unpack_policy(%x : i32) attributes {
    public_name = "bad_f16_unpack_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // expected-error@+1 {{f16 fragment unpack requires packed layout and to_f32 policy}}
    %bad = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f16_pack_policy(%x : f32) attributes {
    public_name = "bad_f16_pack_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // expected-error@+1 {{f16 fragment pack requires packed layout and from_f32 policy}}
    %bad = vc4kernel.fragment_pack %v {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xf32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f16_unpack_result(%x : i32) attributes {
    public_name = "bad_f16_unpack_result",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // expected-error@+1 {{result type must be vector<16xf32>}}
    %bad = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f16_pack_input(%x : i32) attributes {
    public_name = "bad_f16_pack_input",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // expected-error@+1 {{input type must be vector<16xf32>}}
    %bad = vc4kernel.fragment_pack %v {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<from_f32>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_native_f16_alu() attributes {
    public_name = "bad_native_f16_alu",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %v = arith.constant dense<0.000000e+00> : vector<16xf16>
    // expected-error@+1 {{operand #0 must be variadic of vector<16xi32> or vector<16xf32>}}
    %bad = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf16>, vector<16xf16>) -> vector<16xf16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_native_f16_cmp() attributes {
    public_name = "bad_native_f16_cmp",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %v = arith.constant dense<0.000000e+00> : vector<16xf16>
    // expected-error@+1 {{operand #0 must be vector<16xi32> or vector<16xf32>}}
    %bad = vc4kernel.fragment_cmp %v, %v {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf16>, vector<16xf16> -> !vc4kernel.pred<16>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_native_f16_reduce() attributes {
    public_name = "bad_native_f16_reduce",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = arith.constant dense<0.000000e+00> : vector<16xf16>
    // expected-error@+1 {{operand #0 must be vector<16xi32> or vector<16xf32>}}
    %bad = vc4kernel.fragment_reduce %v, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf16>, !vc4kernel.pred<16> -> vector<16xf16>
    vc4kernel.return
  }
}
