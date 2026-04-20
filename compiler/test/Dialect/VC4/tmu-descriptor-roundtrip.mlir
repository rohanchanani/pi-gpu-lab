// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @tmu_descriptor_attrs attributes {part = #vc4.tmu_read_part<rg1616>, unit = #vc4.tmu_unit<tmu1>} {
vc4.module @tmu_descriptor_attrs attributes {unit = #vc4.tmu_unit<tmu1>, part = #vc4.tmu_read_part<rg1616>} {
}

// CHECK: vc4.module @tmu_descriptors {
// CHECK: vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
// CHECK: %[[DIRECT:.*]] = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>} : !vc4.tmu.desc
// CHECK: %[[TEX2D:.*]] = vc4.tmu.descriptor {base = 4096 : i32, height = 64 : i32, mag_filter = #vc4.mag_filter<linear>, min_filter = #vc4.min_filter<lin_mip_lin>, mip_levels = 5 : i32, mode = #vc4.tmu_mode<texture2d>, texture_type = #vc4.texture_type<rgba8888>, width = 128 : i32, wrap_s = #vc4.wrap_mode<repeat>, wrap_t = #vc4.wrap_mode<clamp>} : !vc4.tmu.desc
// CHECK: %[[CUBE:.*]] = vc4.tmu.descriptor {base = 8192 : i32, bias_flags = {per_pixel = true}, child_image_fields = {height = 16 : i32, width = 16 : i32}, cube_map_stride = 256 : i32, flip_y = true, height = 32 : i32, mag_filter = #vc4.mag_filter<nearest>, min_filter = #vc4.min_filter<near_mip_near>, mip_levels = 6 : i32, mode = #vc4.tmu_mode<cubemap>, texture_type = #vc4.texture_type<rgb565>, width = 32 : i32, wrap_s = #vc4.wrap_mode<mirror>, wrap_t = #vc4.wrap_mode<border>} : !vc4.tmu.desc

vc4.module @tmu_descriptors {
  vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %direct = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>} : !vc4.tmu.desc
    %tex2d = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<texture2d>,
      base = 4096 : i32,
      texture_type = #vc4.texture_type<rgba8888>,
      mip_levels = 5 : i32,
      width = 128 : i32,
      height = 64 : i32,
      mag_filter = #vc4.mag_filter<linear>,
      min_filter = #vc4.min_filter<lin_mip_lin>,
      wrap_s = #vc4.wrap_mode<repeat>,
      wrap_t = #vc4.wrap_mode<clamp>
    } : !vc4.tmu.desc
    %cube = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<cubemap>,
      base = 8192 : i32,
      texture_type = #vc4.texture_type<rgb565>,
      mip_levels = 6 : i32,
      width = 32 : i32,
      height = 32 : i32,
      mag_filter = #vc4.mag_filter<nearest>,
      min_filter = #vc4.min_filter<near_mip_near>,
      wrap_s = #vc4.wrap_mode<mirror>,
      wrap_t = #vc4.wrap_mode<border>,
      flip_y = true,
      cube_map_stride = 256 : i32,
      child_image_fields = {width = 16 : i32, height = 16 : i32},
      bias_flags = {per_pixel = true}
    } : !vc4.tmu.desc
    vc4.return
  }
}
