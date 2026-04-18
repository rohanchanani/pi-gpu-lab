// RUN: vc4-opt %s --verify-diagnostics

vc4.module @vpm_read_descriptor_missing_count {
  vc4.func @bad_desc() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<read>, orientation = #vc4.vpm_orientation<horizontal>, lane_mode = #vc4.vpm_lane_mode<packed>, elem_width = #vc4.vpm_elem_width<w32>} : !vc4.vpm.desc // expected-error {{kind = read requires a 'num_vectors' attribute}}
    vc4.return
  }
}

vc4.module @vpm_write_descriptor_extra_count {
  vc4.func @bad_write_desc() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<write>, orientation = #vc4.vpm_orientation<vertical>, lane_mode = #vc4.vpm_lane_mode<laned>, elem_width = #vc4.vpm_elem_width<w16>, num_vectors = 1 : i32} : !vc4.vpm.desc // expected-error {{kind = write must not carry 'num_vectors'}}
    vc4.return
  }
}

vc4.module @vpm_descriptor_addr_range {
  vc4.func @bad_addr() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<write>, orientation = #vc4.vpm_orientation<vertical>, lane_mode = #vc4.vpm_lane_mode<packed>, elem_width = #vc4.vpm_elem_width<w8>, addr = -1 : i32} : !vc4.vpm.desc // expected-error {{'addr' attribute must be non-negative}}
    vc4.return
  }
}

vc4.module @vpm_read_kind_mismatch {
  vc4.func @bad_read(%vin: vector<16xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<write>, orientation = #vc4.vpm_orientation<vertical>, lane_mode = #vc4.vpm_lane_mode<laned>, elem_width = #vc4.vpm_elem_width<w32>} : !vc4.vpm.desc
    %0 = vc4.vpm.read %desc : (!vc4.vpm.desc) -> vector<16xi32> // expected-error {{descriptor kind must be <read>}}
    vc4.return
  }
}

vc4.module @vpm_write_kind_mismatch {
  vc4.func @bad_write(%vin: vector<16xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<read>, orientation = #vc4.vpm_orientation<horizontal>, lane_mode = #vc4.vpm_lane_mode<packed>, elem_width = #vc4.vpm_elem_width<w32>, num_vectors = 1 : i32} : !vc4.vpm.desc
    vc4.vpm.write %desc, %vin : (!vc4.vpm.desc, vector<16xi32>) -> () // expected-error {{descriptor kind must be <write>}}
    vc4.return
  }
}

vc4.module @vpm_write_value_type_error {
  vc4.func @bad_value(%vin: i16) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.vpm.desc {kind = #vc4.vpm_desc_kind<write>, orientation = #vc4.vpm_orientation<horizontal>, lane_mode = #vc4.vpm_lane_mode<packed>, elem_width = #vc4.vpm_elem_width<w32>} : !vc4.vpm.desc
    vc4.vpm.write %desc, %vin : (!vc4.vpm.desc, i16) -> () // expected-error {{value type must be i32, f32, vector<16xi32>, or vector<16xf32>}}
    vc4.return
  }
}
