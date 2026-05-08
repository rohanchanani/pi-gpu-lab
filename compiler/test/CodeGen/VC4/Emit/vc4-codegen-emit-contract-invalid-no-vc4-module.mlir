// RUN: rm -rf %t.bundle
// RUN: not vc4-codegen %s --emit-bundle %t.bundle 2>&1 | FileCheck %s

// CHECK: expected exactly one vc4.module for artifact emission

module {
}
