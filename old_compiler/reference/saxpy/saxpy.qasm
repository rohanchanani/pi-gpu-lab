.include "../share/vc4inc/vc4.qinc"

# Compiler reference kernel for the current gpu->vc4 SAXPY slice.
#
# This is the intended concrete lowering target for:
#   gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index)
#
# Physical uniform stream order for one launched QPU:
#   ra0 = x base address
#   ra1 = y base address
#   ra2 = alpha (one scalar f32 word; uniform reads are lane-broadcast)
#   ra3 = n element count
#   ra4 = qpu_id
#   ra5 = num_qpus
#
# These are consumed sequentially with repeated mov ..., unif. Even though
# qpu_id and num_qpus are physically carried in the uniform stream, they remain
# execution builtins conceptually in the compiler IR.
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif

# Realize the abstract staged rows used by the current lowering:
#   row 0 -> physical row (2 * qpu_id)
#   row 1 -> physical row (2 * qpu_id + 1)
shl r3, ra4, 1
mov ra6, r3
add ra7, r3, 1

# Lowered execution model:
#   base   = qpu_id * 16
#   stride = num_qpus * 16
#
# DMA addresses are byte-addressed, so multiply the element quantities by 4.
shl r0, ra4, 6
mov ra8, r0

shl r3, ra5, 6
mov rb20, r3

shl r1, ra3, 2
mov ra9, r1

mov r0, ra8
add ra10, ra0, r0
add ra11, ra1, r0

# If this QPU's initial chunk base is already >= n, do no work.
mov r1, ra9
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

:loop
    # Stage 16 x values into the physical row for abstract row 0.
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    shl r1, ra6, 4
    add vr_setup, r2, r1
    mov vr_addr, ra10
    mov -, vr_wait

    # Stage 16 y values into the physical row for abstract row 1.
    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    shl r1, ra7, 4
    add vr_setup, r2, r1
    mov vr_addr, ra11
    mov -, vr_wait

    # Read the staged vectors back from VPM.
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

    # Stage the result back into the physical row for abstract row 1.
    mov r3, vpm_setup(1, 1, h32(0))
    add vw_setup, r3, ra7
    mov vpm, r2
    mov -, vw_wait

    # DMA the result vector back to y.
    shl r1, ra7, 7
    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
    add vw_setup, r2, r1
    mov vw_addr, ra11
    mov -, vw_wait

    add ra8, ra8, rb20
    add ra10, ra10, rb20
    add ra11, ra11, rb20

    # Loop while the next chunk base is still < n.
    mov r1, ra9
    sub.setf r1, ra8, r1
    brr.anyc -, :loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
nop
