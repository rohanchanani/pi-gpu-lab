// RUN: rm -rf %t.single %t.multi
// RUN: vc4-codegen %S/../Hardware/Run/saxpy_full/input.mlir --emit-bundle %t.single
// RUN: vc4-codegen %S/../Hardware/Run/multi_kernel_minimal/input.mlir --emit-bundle %t.multi
// RUN: test -f %t.single/layout.json
// RUN: test -f %t.multi/layout.json
// RUN: python3 %S/../Support/check_m2_layout.py %t.single/layout.json --alignment 8
// RUN: python3 %S/../Support/check_m2_layout.py %t.multi/layout.json --alignment 8

// M2 layout smoke: code/uniform/unif_ptr regions must be persistent,
// 8-byte-aligned, and non-overlapping. Heap is introduced in m2-04.
