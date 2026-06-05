// RUN: python3 %S/../../../../test/Conversion/SSAVC4ToVC4/Support/audit_branch_layout_accounting.py %S/../../../../lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp | FileCheck %s

// CHECK: branch layout accounting audit PASS
// CHECK: allowed fixed constants:
