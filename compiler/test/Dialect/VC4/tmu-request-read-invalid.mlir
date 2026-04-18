// RUN: vc4-opt %s --verify-diagnostics

vc4.module @direct_address_type_error {
  vc4.func @bad_direct(%addr: f32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (f32) -> !vc4.async.token // expected-error {{direct-mode address operand must be i32 or vector<16xi32>}}
    vc4.return
  }
}

vc4.module @descriptor_last_error {
  vc4.func @bad_order(%addr: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>} : !vc4.tmu.desc
    %0 = "vc4.tmu.request"(%desc, %addr) <{unit = #vc4.tmu_unit<tmu0>}> : (!vc4.tmu.desc, i32) -> !vc4.async.token // expected-error {{descriptor operand must be the last operand}}
    vc4.return
  }
}

vc4.module @cubemap_operand_count_error {
  vc4.func @bad_cube(%s: f32, %t: f32) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<cubemap>,
      texture_type = #vc4.texture_type<rgb565>,
      width = 16 : i32,
      height = 16 : i32,
      cube_map_stride = 64 : i32
    } : !vc4.tmu.desc
    %0 = "vc4.tmu.request"(%s, %t, %desc) <{unit = #vc4.tmu_unit<tmu1>}> : (f32, f32, !vc4.tmu.desc) -> !vc4.async.token // expected-error {{expects between 3 and 4 coordinate operands for the selected TMU mode}}
    vc4.return
  }
}

vc4.module @read_token_unit_error {
  vc4.func @bad_read_unit(%addr: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %tok = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
    %0 = "vc4.tmu.read"(%tok) <{unit = #vc4.tmu_unit<tmu1>, part = #vc4.tmu_read_part<raw32>}> : (!vc4.async.token) -> i32 // expected-error {{token unit must match the selected read unit}}
    vc4.return
  }
}

vc4.module @read_packed_type_error {
  vc4.func @bad_read_type() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.tmu.read"() <{unit = #vc4.tmu_unit<tmu0>, part = #vc4.tmu_read_part<rgba8888>}> : () -> f32 // expected-error {{packed TMU read parts require i32 or vector<16xi32> result type}}
    vc4.return
  }
}

vc4.module @noswap_mode_selection_error {
  vc4.func @bad_noswap(%flag: i1) attributes {threading = 0 : i32, form = 0 : i32} {
    "vc4.tmu.noswap"(%flag) <{disable = false}> : (i1) -> () // expected-error {{requires exactly one of a value operand or a 'disable' attribute}}
    vc4.return
  }
}

vc4.module @noswap_missing_form_error {
  vc4.func @bad_noswap_missing() attributes {threading = 0 : i32, form = 0 : i32} {
    "vc4.tmu.noswap"() : () -> () // expected-error {{requires exactly one of a value operand or a 'disable' attribute}}
    vc4.return
  }
}
