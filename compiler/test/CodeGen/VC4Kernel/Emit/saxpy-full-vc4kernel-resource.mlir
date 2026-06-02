// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %S/../Hardware/Run/saxpy_full_vc4kernel/input.mlir --verify-vc4kernel -o %t/saxpy.verified.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VERIFY --input-file=%t/saxpy.verified.vc4kernel.mlir
// RUN: vc4-opt %t/saxpy.verified.vc4kernel.mlir --convert-vc4kernel-to-ssavc4 -o %t/saxpy.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/saxpy.ssavc4.mlir
// RUN: vc4-opt %t/saxpy.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/saxpy.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/saxpy.vc4.mlir
// RUN: vc4-codegen %t/saxpy.vc4.mlir --emit-bundle %t/bundle
// RUN: test -f %t/bundle/kernel_launch.c
// RUN: test -f %t/bundle/manifest.json
// RUN: test -f %t/bundle/kernels/saxpy_full_vc4kernel.qasm
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t/bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t/bundle/manifest.json
// RUN: python3 -c "from pathlib import Path; import re, sys; s=Path('%t/bundle/kernel_launch.c').read_text()+Path('%t/bundle/manifest.json').read_text(); bad=['shared_'+'vpm_'+'bytes','user_shared_'+'vpm_'+'rows_'+'per_block','vpm_'+'rows_'+'per_block','vpm_'+'bytes_'+'per_block','warps_per_'+'block_max','uses_shared_'+'vpm','require_full_block_'+'residency','semaphores_'+'per_block']; hit=[x for x in bad if re.search(r'(?<![A-Za-z0-9_])'+re.escape(x)+r'(?![A-Za-z0-9_])', s)]; sys.exit('old resource fields in generated bundle: '+','.join(hit) if hit else 0)"

// VERIFY: vc4kernel.kernel @saxpy_full_vc4kernel
// VERIFY: vc4kernel.program_id
// VERIFY: vc4kernel.pred.tail
// VERIFY: vc4kernel.tmu_load_fragment
// VERIFY: vc4kernel.vdw_store_fragment
// VERIFY-NOT: ssavc4.

// SSAVC4-LABEL: ssavc4.func @saxpy_full_vc4kernel
// SSAVC4-SAME: {direction = "in", elem_type = "f32", kind = "buffer", name = "x", uniform_index = 0 : i32}
// SSAVC4-SAME: {direction = "inout", elem_type = "f32", kind = "buffer", name = "y", uniform_index = 1 : i32}
// SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "alpha", type = "f32", uniform_index = 2 : i32}
// SSAVC4-SAME: {direction = "by_value", kind = "scalar", name = "n", type = "i32", uniform_index = 3 : i32}
// SSAVC4-SAME: {kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", name = "logical_request", uniform_index = 4 : i32}
// SSAVC4-SAME: {kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", name = "total_requests", uniform_index = 5 : i32}
// SSAVC4-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 6 : i32}
// SSAVC4-SAME: vc4.resource
// SSAVC4: compiler_vpm_staging_rows_per_warp = 1 : i32
// SSAVC4: requires_vpm_base_row_builtin = true
// SSAVC4: total_vpm_rows_per_block = 1 : i32
// SSAVC4: uses_tmu = true
// SSAVC4: uses_vdw = true
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store
// SSAVC4-NOT: vc4kernel.
// SSAVC4-NOT: QPU_NUMBER

// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.

// SOURCE: .name = "saxpy_full_vc4kernel"
// SOURCE: .resource = {
// SOURCE: .schedule_mode = VC4_SCHEDULE_INDEPENDENT_VECTOR
// SOURCE: .warps_per_block = KERNEL_0_WARPS_PER_BLOCK
// SOURCE: .uses_tmu = 1u
// SOURCE: .uses_vdw = 1u
// SOURCE: .user_vpm_rows_per_block = KERNEL_0_USER_VPM_ROWS_PER_BLOCK
// SOURCE: .compiler_vpm_staging_rows_per_warp = KERNEL_0_COMPILER_VPM_STAGING_ROWS_PER_WARP
// SOURCE: .total_vpm_rows_per_block = KERNEL_0_TOTAL_VPM_ROWS_PER_BLOCK
// SOURCE: .requires_vpm_base_row_builtin = 1u
// SOURCE: [4] builtin logical_request
// SOURCE: [5] builtin total_requests
// SOURCE: [6] builtin vpm_base_row

// MANIFEST: "public_name": "saxpy_full_vc4kernel"
// MANIFEST: "uniform_words_per_request": 7
// MANIFEST: "resources":
// MANIFEST: "compiler_vpm_staging_rows_per_warp": 1
// MANIFEST: "total_vpm_rows_per_block": 1
// MANIFEST: "uses_tmu": true
// MANIFEST: "uses_vdw": true
// MANIFEST: "requires_vpm_base_row_builtin": true
