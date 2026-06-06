// RUN: not vc4-opt %s --split-input-file -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @bad_vpm_w16 {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: sub-32 VPM QPU access requires subword = #ssavc4.vpm_subword<packed> or #ssavc4.vpm_subword<laned>
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_w8 {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: sub-32 VPM QPU access requires subword = #ssavc4.vpm_subword<packed> or #ssavc4.vpm_subword<laned>
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_packed {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: 32-bit VPM QPU access requires subword = #ssavc4.vpm_subword<none>
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<packed>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_laned {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: 32-bit VPM QPU access requires subword = #ssavc4.vpm_subword<none>
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<laned>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_vertical_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: requires VPM x coordinate in range [0, 15]
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 16 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_horizontal_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: horizontal 32-bit VPM QPU access requires x = 0
    ssavc4.vpm.write %row, %value {orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 1 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_string_orientation {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK-DAG: attribute 'orientation' failed to satisfy constraint
    ssavc4.vpm.write %row, %value {orientation = "horizontal", width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
