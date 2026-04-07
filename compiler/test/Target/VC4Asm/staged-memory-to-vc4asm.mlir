// Translation smoke test for staged-memory VC4 ops plus vector compute.
// RUN: vc4-translate --mlir-to-vc4asm %s | FileCheck %s

module {
  %xrow = arith.constant 0 : index
  %yrow = arith.constant 1 : index
  %offset = arith.constant 0 : index
  %x = vc4.get_uniform[0] : memref<?xf32>
  %y = vc4.get_uniform[1] : memref<?xf32>
  %a = vc4.get_uniform[2] : vector<16xf32>

  vc4.dma_load %x[%offset] to %xrow : memref<?xf32>, index, index
  vc4.dma_load %y[%offset] to %yrow : memref<?xf32>, index, index
  %xv = vc4.vpm_read from %xrow : index -> vector<16xf32>
  %yv = vc4.vpm_read from %yrow : index -> vector<16xf32>
  %mul = vc4.fmul %a, %xv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  %sum = vc4.fadd %mul, %yv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  vc4.vpm_write %sum to %yrow : vector<16xf32>, index
  vc4.dma_store %yrow to %y[%offset] : index, memref<?xf32>, index
}

// CHECK: ; vc4asm-like output generated from staged vc4 IR
// CHECK: ; contract: schematic printer only, not executable codegen
// CHECK: ; placeholders are marked explicitly where exact vc4asm syntax is TBD
// CHECK: ; Uniform reads
// CHECK: ; uniform_0 = vc4.get_uniform[0] : memref<?xf32>
// CHECK: mov uniform_0, unif
// CHECK: ; uniform_1 = vc4.get_uniform[1] : memref<?xf32>
// CHECK: mov uniform_1, unif
// CHECK: ; uniform_2 = vc4.get_uniform[2] : vector<16xf32>
// CHECK: mov uniform_2, unif
// CHECK: ; DMA staging
// CHECK: ; vc4.dma_load uniform_0[0] to row 0
// CHECK: ; schematic vc4asm for global-memory -> VPM staging
// CHECK: ;   mov vr_setup, vdr_setup_0(... row=0 ...)
// CHECK: ;   mov vr_addr, uniform_0 + 0
// CHECK: ;   mov -, vr_wait
// CHECK: ; vc4.dma_load uniform_1[0] to row 1
// CHECK: ; schematic vc4asm for global-memory -> VPM staging
// CHECK: ;   mov vr_setup, vdr_setup_0(... row=1 ...)
// CHECK: ;   mov vr_addr, uniform_1 + 0
// CHECK: ;   mov -, vr_wait
// CHECK: ; VPM access
// CHECK: ; tmp0 = vc4.vpm_read from row 0
// CHECK: ; schematic vc4asm for VPM -> QPU vector read
// CHECK: ;   mov tmp0, vpm
// CHECK: ; tmp1 = vc4.vpm_read from row 1
// CHECK: ; schematic vc4asm for VPM -> QPU vector read
// CHECK: ;   mov tmp1, vpm
// CHECK: ; Compute
// CHECK: ; tmp2 = vc4.fmul uniform_2, tmp0
// CHECK: fmul tmp2, uniform_2, tmp0
// CHECK: ; tmp3 = vc4.fadd tmp2, tmp1
// CHECK: fadd tmp3, tmp2, tmp1
// CHECK: ; VPM access
// CHECK: ; vc4.vpm_write tmp3 to row 1
// CHECK: ; schematic vc4asm for QPU vector -> VPM write
// CHECK: ;   mov vpm, tmp3
// CHECK: ; DMA staging
// CHECK: ; vc4.dma_store row 1 to uniform_1[0]
// CHECK: ; schematic vc4asm for VPM -> global-memory staging
// CHECK: ;   mov vw_setup, vdw_setup_0(... row=1 ...)
// CHECK: ;   mov vw_addr, uniform_1 + 0
// CHECK: ;   mov -, vw_wait
