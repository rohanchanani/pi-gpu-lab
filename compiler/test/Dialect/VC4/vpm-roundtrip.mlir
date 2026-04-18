// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @vpm_ops {
// CHECK: vc4.func @main(%[[VIN:.*]]: vector<16xi32>, %[[SCALAR:.*]]: f32) attributes {form = 0 : i32, threading = 0 : i32} {
// CHECK: %[[READ_DESC:.*]] = vc4.vpm.desc {addr = 0 : i32, elem_width = #vc4.vpm_elem_width<w32>, kind = #vc4.vpm_desc_kind<read>, lane_mode = #vc4.vpm_lane_mode<packed>, num_vectors = 2 : i32, orientation = #vc4.vpm_orientation<horizontal>, stride = 1 : i32} : !vc4.vpm.desc
// CHECK: %[[WRITE_DESC:.*]] = vc4.vpm.desc {addr = 16 : i32, elem_width = #vc4.vpm_elem_width<w16>, kind = #vc4.vpm_desc_kind<write>, lane_mode = #vc4.vpm_lane_mode<laned>, orientation = #vc4.vpm_orientation<vertical>, stride = 4 : i32} : !vc4.vpm.desc
// CHECK: %[[R0:.*]] = vc4.vpm.read %[[READ_DESC]] : (!vc4.vpm.desc) -> vector<16xi32>
// CHECK: vc4.vpm.write %[[WRITE_DESC]], %[[VIN]] : (!vc4.vpm.desc, vector<16xi32>) -> ()
// CHECK: vc4.vpm.write %[[WRITE_DESC]], %[[SCALAR]] : (!vc4.vpm.desc, f32) -> ()
// CHECK: vc4.return

vc4.module @vpm_ops {
  vc4.func @main(%vin: vector<16xi32>, %scalar: f32) attributes {threading = 0 : i32, form = 0 : i32} {
    %read_desc = vc4.vpm.desc {
      kind = #vc4.vpm_desc_kind<read>,
      orientation = #vc4.vpm_orientation<horizontal>,
      lane_mode = #vc4.vpm_lane_mode<packed>,
      elem_width = #vc4.vpm_elem_width<w32>,
      addr = 0 : i32,
      stride = 1 : i32,
      num_vectors = 2 : i32
    } : !vc4.vpm.desc
    %write_desc = vc4.vpm.desc {
      kind = #vc4.vpm_desc_kind<write>,
      orientation = #vc4.vpm_orientation<vertical>,
      lane_mode = #vc4.vpm_lane_mode<laned>,
      elem_width = #vc4.vpm_elem_width<w16>,
      addr = 16 : i32,
      stride = 4 : i32
    } : !vc4.vpm.desc

    %r0 = vc4.vpm.read %read_desc : (!vc4.vpm.desc) -> vector<16xi32>
    vc4.vpm.write %write_desc, %vin : (!vc4.vpm.desc, vector<16xi32>) -> ()
    vc4.vpm.write %write_desc, %scalar : (!vc4.vpm.desc, f32) -> ()
    vc4.return
  }
}
