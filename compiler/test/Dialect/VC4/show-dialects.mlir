// RUN: vc4-opt --show-dialects %s 2>&1 | FileCheck %s

// CHECK: Available Dialects:
// CHECK-SAME: vc4

module {
}
