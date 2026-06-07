// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_bf16_subword(%x : i32) attributes {
    public_name = "bad_bf16_subword",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: expected ::mlir::vc4kernel::SubwordType to be one of: u8, s8, u16, s16, f16
    %bad = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<bf16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_fp8_subword(%x : i32) attributes {
    public_name = "bad_fp8_subword",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: expected ::mlir::vc4kernel::SubwordType to be one of: u8, s8, u16, s16, f16
    %bad = vc4kernel.fragment_unpack %v {source = #vc4kernel.subword_type<fp8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}
