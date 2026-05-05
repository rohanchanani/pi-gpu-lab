// RUN: rm -rf %t.bundle
// RUN: not vc4-codegen %s --emit-bundle %t.bundle 2>&1 | FileCheck %s

module {
}

// CHECK: vc4-codegen requires exactly one eligible kernel
