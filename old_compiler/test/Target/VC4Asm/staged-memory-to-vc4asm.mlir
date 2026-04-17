// Translation smoke test for abstract staged-memory VC4 ops plus vector compute.
// RUN: vc4-translate --mlir-to-vc4asm %s | FileCheck %s

module {
  %offset = arith.constant 0 : index
  %x = vc4.get_uniform[0] : memref<?xf32>
  %y = vc4.get_uniform[1] : memref<?xf32>
  %a = vc4.get_uniform[2] : vector<16xf32>

  vc4.dma_load %x[%offset] to slot[2] : memref<?xf32>, index
  vc4.dma_load %y[%offset] to slot[3] : memref<?xf32>, index
  %xv = vc4.staged_read from slot[2] -> vector<16xf32>
  %yv = vc4.staged_read from slot[3] -> vector<16xf32>
  %mul = vc4.fmul %a, %xv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  %sum = vc4.fadd %mul, %yv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  vc4.staged_write slot[5] = %sum : vector<16xf32>
  vc4.dma_store slot[5] to %y[%offset] : memref<?xf32>, index
}

// CHECK: ; inspection-only VC4 assembly sketch generated from lowered vc4 IR
// CHECK: ; contract: intermediate inspection output, not the final artifact boundary
// CHECK: ; placeholders are marked explicitly where exact VC4 assembly syntax is still TBD
// CHECK: ; Uniform reads
// CHECK: ; uniform_0 = vc4.get_uniform[0] : memref<?xf32>
// CHECK: mov uniform_0, unif
// CHECK: ; uniform_1 = vc4.get_uniform[1] : memref<?xf32>
// CHECK: mov uniform_1, unif
// CHECK: ; uniform_2 = vc4.get_uniform[2] : vector<16xf32>
// CHECK: mov uniform_2, unif
// CHECK: ; DMA staging
// CHECK: ; vc4.dma_load uniform_0[0] to slot[2]
// CHECK: ; debug-only emitter-local mapping: slot[2] currently prints as physical row
// CHECK: ; inspection sketch for global-memory -> VPM staging
// CHECK: ;   mov vr_setup, vdr_setup_0(... row=2 ...)
// CHECK: ;   mov vr_addr, uniform_0 + 0
// CHECK: ;   mov -, vr_wait
// CHECK: ; vc4.dma_load uniform_1[0] to slot[3]
// CHECK: ; debug-only emitter-local mapping: slot[3] currently prints as physical row
// CHECK: ; inspection sketch for global-memory -> VPM staging
// CHECK: ;   mov vr_setup, vdr_setup_0(... row=3 ...)
// CHECK: ;   mov vr_addr, uniform_1 + 0
// CHECK: ;   mov -, vr_wait
// CHECK: ; Staging access
// CHECK: ; tmp0 = vc4.staged_read from slot[2]
// CHECK: ; debug-only emitter-local mapping: slot[2] currently prints as physical row
// CHECK: ; inspection sketch for VPM -> QPU vector read
// CHECK: ;   mov tmp0, vpm
// CHECK: ; tmp1 = vc4.staged_read from slot[3]
// CHECK: ; debug-only emitter-local mapping: slot[3] currently prints as physical row
// CHECK: ; inspection sketch for VPM -> QPU vector read
// CHECK: ;   mov tmp1, vpm
// CHECK: ; Compute
// CHECK: ; tmp2 = vc4.fmul uniform_2, tmp0
// CHECK: fmul tmp2, uniform_2, tmp0
// CHECK: ; tmp3 = vc4.fadd tmp2, tmp1
// CHECK: fadd tmp3, tmp2, tmp1
// CHECK: ; Staging access
// CHECK: ; vc4.staged_write slot[5] = tmp3
// CHECK: ; debug-only emitter-local mapping: slot[5] currently prints as physical row
// CHECK: ; inspection sketch for QPU vector -> VPM write
// CHECK: ;   mov vpm, tmp3
// CHECK: ; DMA staging
// CHECK: ; vc4.dma_store slot[5] to uniform_1[0]
// CHECK: ; debug-only emitter-local mapping: slot[5] currently prints as physical row
// CHECK: ; inspection sketch for VPM -> global-memory staging
// CHECK: ;   mov vw_setup, vdw_setup_0(... row=5 ...)
// CHECK: ;   mov vw_addr, uniform_1 + 0
// CHECK: ;   mov -, vw_wait
