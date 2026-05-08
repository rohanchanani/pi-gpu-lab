// RUN: rm -rf %t.bundle
// RUN: not vc4-codegen %s --emit-bundle %t.bundle 2>&1 | FileCheck %s

// CHECK: expected exactly one eligible VC4 QPU kernel

vc4.module @vc4_codegen_skeleton_invalid {
}
