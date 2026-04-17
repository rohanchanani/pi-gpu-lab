.include "share/vc4inc/vc4.qinc"

# qasm emitted from lowered vc4 kernel model
# kernel @saxpy

# uniform[0] = arg0
mov ra0, unif
# uniform[1] = arg1
mov ra1, unif
# uniform[2] = arg2
mov ra2, unif
# uniform[3] = arg3
mov ra3, unif
# uniform[4] = builtin qpu_id
mov ra4, unif
# uniform[5] = builtin num_qpus
mov ra5, unif

# base = qpu_id * 16 elements; stride = num_qpus * 16 elements
# qasm uses byte-addressed DMA pointers, so the loop offset is scaled by 4
shl r0, ra4, 6
mov ra6, r0
shl r0, ra5, 6
mov ra7, r0
shl r0, ra3, 2
mov ra8, r0

# Skip execution if this worker's initial chunk is already out of range
mov r1, ra8
sub.setf r1, ra6, r1
brr.anync -, :end
nop
nop
nop

:loop
    # dma_load uniform0[iv] -> stage0 (emitter-local row 0)
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    mov vr_setup, r2
    add vr_addr, ra0, ra6
    mov -, vr_wait

    # staged_read stage0 -> rb0 (emitter-local row 0)
    mov r2, vpm_setup(1, 1, h32(0))
    mov vr_setup, r2
    mov rb0, vpm
    mov -, vw_wait

    # dma_load uniform1[iv] -> stage1 (emitter-local row 1)
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    add vr_setup, r2, 16
    add vr_addr, ra1, ra6
    mov -, vr_wait

    # staged_read stage1 -> rb1 (emitter-local row 1)
    mov r2, vpm_setup(1, 1, h32(0))
    add vr_setup, r2, 1
    mov rb1, vpm
    mov -, vw_wait

    fmul rb2, ra2, rb0
    fadd rb3, rb2, rb1
    # staged_write rb3 -> stage2 (emitter-local row 2)
    mov r2, vpm_setup(1, 1, h32(0))
    add vw_setup, r2, 2
    mov vpm, rb3
    mov -, vw_wait

    # dma_store stage2 -> uniform1[iv] (emitter-local row 2)
    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
    add vw_setup, r2, 256
    add vw_addr, ra1, ra6
    mov -, vw_wait

    add ra6, ra6, ra7
    mov r1, ra8
    sub.setf r1, ra6, r1
    brr.anyc -, :loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
nop
