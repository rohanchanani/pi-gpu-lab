// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  %carrier = ssavc4.load_imm <splat32> {value = 15360 : i32} : vector<16xi32>
  // expected-error@+1 {{f16 storage conversion unpack requires regfile-A mode f16a_or_i16a}}
  %bad = ssavc4.unpack %carrier {f16_storage_conversion, mode = #vc4.regfile_a_unpack_mode<color8a>} : vector<16xi32> -> vector<16xf32>
}

// ----

module {
  %f = ssavc4.load_imm <splat32> {value = 0.000000e+00 : f32} : vector<16xf32>
  // expected-error@+1 {{f16 storage conversion pack requires regfile-A mode to_16a}}
  %bad = ssavc4.pack %f {f16_storage_conversion, mode = #vc4.regfile_a_pack_mode<to_8a>} : vector<16xf32> -> vector<16xi32>
}

// ----

module {
  %carrier = ssavc4.load_imm <splat32> {value = 15360 : i32} : vector<16xi32>
  // expected-error@+1 {{result type must be an SSAVC4 float carrier}}
  %bad = ssavc4.unpack %carrier {f16_storage_conversion, mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xi32>
}

// ----

module {
  %carrier = ssavc4.load_imm <splat32> {value = 15360 : i32} : vector<16xi32>
  // expected-error@+1 {{input and result must have the same SSAVC4 element domain}}
  %bad = ssavc4.unpack %carrier {mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xf32>
}
