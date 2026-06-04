// RUN: python3 %S/../../../../test/Conversion/SSAVC4ToVC4/Support/audit_spill_edgecopy_slot_accounting.py %S/../../../../lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp | FileCheck %s

// CHECK: spill/edge-copy slot accounting audit PASS
