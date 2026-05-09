// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %S/../Hardware/Run/saxpy_full/input.mlir --emit-bundle %t.bundle
// RUN: python3 %S/../Support/check_m2_manifest.py %t.bundle/manifest.json --schema-version 2 --kernel-count 1 --require-target --require-kernel-fields --public-name saxpy_full
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h

// M2 manifest-v2 regression for the one-kernel case. A single kernel is a
// program bundle with kernels.length == 1, not a separate legacy schema.
