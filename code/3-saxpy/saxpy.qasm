.include "../share/vc4inc/vc4.qinc"

# Uniform layout:
#   ra0 = x base address
#   ra1 = y base address
#   ra2 = alpha
#   ra3 = num qpus
#   ra4 = qpu index
#   ra5 = iteration count
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif

# Reserve two VPM rows per QPU: one for X and one for Y / result.
shl r3, ra4, 1
mov ra6, r3
add ra7, r3, 1

# Initial byte offset for this QPU: qpu * 16 elements * 4 bytes.
shl r0, ra4, 6
add ra8, ra0, r0
add ra9, ra1, r0

# Stride between iterations: num_qpus * 16 elements * 4 bytes.
shl r3, ra3, 6
mov ra10, r3
mov rb20, r3

mov ra11, ra5

:loop
    # DMA read 16 X values into VPM row ra6.
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    shl r1, ra6, 4
    add vr_setup, r2, r1
    mov vr_addr, ra8
    mov -, vr_wait

    # DMA read 16 Y values into VPM row ra7.
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    shl r1, ra7, 4
    add vr_setup, r2, r1
    mov vr_addr, ra9
    mov -, vr_wait

    # Load the two vectors back from VPM into accumulators.
    mov r2, vpm_setup(1, 1, h32(0))

    add vr_setup, r2, ra6
    mov r0, vpm
    mov -, vw_wait

    add vr_setup, r2, ra7
    mov r1, vpm
    mov -, vw_wait

    # y = alpha * x + y
    fmul r2, r0, ra2
    fadd r2, r2, r1

    # Write the result vector into the Y row and DMA it back to memory.
    mov r3, vpm_setup(1, 1, h32(0))
    add vw_setup, r3, ra7
    mov vpm, r2
    mov -, vw_wait

    shl r1, ra7, 7
    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
    add vw_setup, r2, r1
    mov vw_addr, ra9
    mov -, vw_wait

    add ra8, ra8, rb20
    add ra9, ra9, rb20

    sub.setf ra11, ra11, 1
    brr.anynz -, :loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
nop
