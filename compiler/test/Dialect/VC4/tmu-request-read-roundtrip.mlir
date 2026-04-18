// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @tmu_request_read {
// CHECK: %[[DIRECT_DESC:.*]] = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>} : !vc4.tmu.desc
// CHECK: %[[TEX_DESC:.*]] = vc4.tmu.descriptor {height = 64 : i32, mode = #vc4.tmu_mode<texture2d>, texture_type = #vc4.texture_type<rgba8888>, width = 64 : i32} : !vc4.tmu.desc
// CHECK: %[[CUBE_DESC:.*]] = vc4.tmu.descriptor {cube_map_stride = 128 : i32, height = 32 : i32, mode = #vc4.tmu_mode<cubemap>, texture_type = #vc4.texture_type<rgb565>, width = 32 : i32} : !vc4.tmu.desc
// CHECK: %[[TOK0:.*]] = "vc4.tmu.request"(%[[ADDR:.*]]) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
// CHECK: %[[TOK1:.*]] = "vc4.tmu.request"(%[[ADDR]], %[[DIRECT_DESC]]) <{unit = #vc4.tmu_unit<tmu0>}> : (i32, !vc4.tmu.desc) -> !vc4.async.token
// CHECK: "vc4.tmu.request"(%[[S0:.*]], %[[T0:.*]], %[[TEX_DESC]]) <{unit = #vc4.tmu_unit<tmu1>}> : (f32, f32, !vc4.tmu.desc) -> ()
// CHECK: %[[TOK2:.*]] = "vc4.tmu.request"(%[[R0:.*]], %[[R0]], %[[R0]], %[[BIAS:.*]], %[[CUBE_DESC]]) <{unit = #vc4.tmu_unit<tmu1>}> : (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, !vc4.tmu.desc) -> !vc4.async.token
// CHECK: %[[RAW:.*]] = "vc4.tmu.read"(%[[TOK0]]) <{part = #vc4.tmu_read_part<raw32>, unit = #vc4.tmu_unit<tmu0>}> : (!vc4.async.token) -> vector<16xf32>
// CHECK: %[[PACKED:.*]] = "vc4.tmu.read"() <{part = #vc4.tmu_read_part<rgba8888>, unit = #vc4.tmu_unit<tmu1>}> : () -> i32
// CHECK: "vc4.tmu.noswap"() <{disable = true}> : () -> ()
// CHECK: "vc4.tmu.noswap"(%[[FLAG:.*]]) : (i1) -> ()

vc4.module @tmu_request_read {
  vc4.func @main(%addr: i32, %s: f32, %t: f32,
                 %r: vector<16xf32>, %bias: vector<16xf32>, %flag: i1) attributes {threading = 0 : i32, form = 0 : i32} {
    %direct_desc = vc4.tmu.descriptor {mode = #vc4.tmu_mode<direct>} : !vc4.tmu.desc
    %tex_desc = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<texture2d>,
      texture_type = #vc4.texture_type<rgba8888>,
      width = 64 : i32,
      height = 64 : i32
    } : !vc4.tmu.desc
    %cube_desc = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<cubemap>,
      texture_type = #vc4.texture_type<rgb565>,
      width = 32 : i32,
      height = 32 : i32,
      cube_map_stride = 128 : i32
    } : !vc4.tmu.desc

    %tok0 = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
    %tok1 = "vc4.tmu.request"(%addr, %direct_desc) <{unit = #vc4.tmu_unit<tmu0>}> : (i32, !vc4.tmu.desc) -> !vc4.async.token
    "vc4.tmu.request"(%s, %t, %tex_desc) <{unit = #vc4.tmu_unit<tmu1>}> : (f32, f32, !vc4.tmu.desc) -> ()
    %tok2 = "vc4.tmu.request"(%r, %r, %r, %bias, %cube_desc) <{unit = #vc4.tmu_unit<tmu1>}> : (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, !vc4.tmu.desc) -> !vc4.async.token

    %raw = "vc4.tmu.read"(%tok0) <{unit = #vc4.tmu_unit<tmu0>, part = #vc4.tmu_read_part<raw32>}> : (!vc4.async.token) -> vector<16xf32>
    %packed = "vc4.tmu.read"() <{unit = #vc4.tmu_unit<tmu1>, part = #vc4.tmu_read_part<rgba8888>}> : () -> i32

    "vc4.tmu.noswap"() <{disable = true}> : () -> ()
    "vc4.tmu.noswap"(%flag) : (i1) -> ()

    vc4.return
  }
}
