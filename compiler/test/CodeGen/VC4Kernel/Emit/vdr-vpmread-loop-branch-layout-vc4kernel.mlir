// RUN: mkdir -p %t
// RUN: vc4-opt %S/../Hardware/Run/vdr_vpmread_loop_pingpong_rows_vc4kernel/input.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 > %t/lowered.vc4.mlir
// RUN: vc4-codegen %t/lowered.vc4.mlir --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/vdr_vpmread_loop_pingpong_rows_vc4kernel.qasm

// QASM: {{^}}:vc4_qpu_slot_40{{$}}
// QASM: sub.setf
// QASM: mov {{ra[0-9]+}}, vpm
// QASM: read vr_wait
// QASM: mov {{ra[0-9]+}}, vpm
// QASM: read vr_wait
// QASM: mov ra11, ra21
// QASM: mov ra10, ra19
// QASM: mov ra13, ra20
// QASM: brr ra31, rb30, -, :vc4_qpu_slot_40 # qpu.branch {{.*}}target_label=vc4_qpu_slot_40
