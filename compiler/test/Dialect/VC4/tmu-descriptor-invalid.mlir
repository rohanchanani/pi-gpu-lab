// RUN: vc4-opt %s --verify-diagnostics

vc4.module @direct_texture_fields_error {
  vc4.func @bad_direct() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>, texture_type = #vc4.texture_type<rgba8888>} : !vc4.tmu.desc // expected-error {{direct mode must not carry texture setup attributes}}
    vc4.return
  }
}

vc4.module @texture2d_cubemap_stride_error {
  vc4.func @bad_texture2d() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.tmu.descriptor {mode = #vc4.tmu_mode<texture2d>, cube_map_stride = 16 : i32} : !vc4.tmu.desc // expected-error {{'cube_map_stride' is only legal for mode = cubemap}}
    vc4.return
  }
}

vc4.module @width_height_pair_error {
  vc4.func @bad_size() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.tmu.descriptor {mode = #vc4.tmu_mode<cubemap>, width = 64 : i32} : !vc4.tmu.desc // expected-error {{requires 'width' and 'height' to be provided together}}
    vc4.return
  }
}

vc4.module @positive_integer_error {
  vc4.func @bad_mips() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.tmu.descriptor {mode = #vc4.tmu_mode<texture2d>, mip_levels = 0 : i32} : !vc4.tmu.desc // expected-error {{'mip_levels' attribute must be greater than zero}}
    vc4.return
  }
}

vc4.module @empty_child_dict_error {
  vc4.func @bad_child_fields() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.tmu.descriptor {mode = #vc4.tmu_mode<cubemap>, child_image_fields = {}} : !vc4.tmu.desc // expected-error {{'child_image_fields' attribute must not be empty}}
    vc4.return
  }
}
