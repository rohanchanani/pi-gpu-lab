// REQUIRES: vc4-has-triton-cpp-frontend

// RUN: python3 %vc4_repo_root/compiler/test/Conversion/TritonToVC4Value/Support/audit_ttir_value_output_boundary.py --repo-root %vc4_repo_root --vc4-triton-opt %vc4_triton_opt | FileCheck %s

// CHECK: TTIR_VALUE_OUTPUT_BOUNDARY_AUDIT=PASS
// CHECK: CONTROLLED_TTIR_VALUE_OUTPUT_BOUNDARY=PASS
// CHECK: READY_FOR_TRITON=NO
