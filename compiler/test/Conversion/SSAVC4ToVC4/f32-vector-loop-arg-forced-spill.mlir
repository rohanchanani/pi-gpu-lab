// RUN: vc4-opt %S/../../../../test/CodeGen/SSAVC4/Hardware/Run/f32_vector_loop_arg_forced_spill_ssavc4/input.mlir --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @f32_vector_loop_arg_forced_spill_ssavc4_kernel
// CHECK-SAME: spill_frame_bytes = {{[1-9][0-9]*}} : i32
// CHECK-SAME: spill_frame_stride_bytes = {{[1-9][0-9]*}} : i32
// CHECK-SAME: spill_frame_base
// CHECK-SAME: spill_vpm_row
// CHECK-DAG: uses_vdr = true
// CHECK-DAG: uses_vpm_qpu_read = true
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>
// CHECK: sig = #vc4.qpu_signal<thrend>
