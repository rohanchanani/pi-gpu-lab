.include "../share/vc4inc/vc4.qinc"

# saxpy_16 reference kernel.
#
# This is the runtime-N TMU-backed SAXPY golden:
#   - loads use direct TMU0 memory lookups
#   - stores use VPM -> VDW
#   - each QPU processes byte offsets qpu_id*64, qpu_id*64 + num_qpus*64, ...
#   - the handwritten harness supplies n as a nonzero multiple of active_qpus*16
#   - no launcher-side n guard and no tail masking yet
#
# Physical uniform stream order for one launched QPU:
#   ra0 = x base address
#   ra1 = y base address/result address
#   ra2 = alpha (one scalar f32 word; uniform reads are lane-broadcast)
#   ra3 = n element count
#   ra4 = qpu_id
#   ra5 = num_qpus
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif

# byte_offset = qpu_id * 16 lanes * sizeof(f32)
shl r0, ra4, 6
mov ra8, r0

# stride_bytes = num_qpus * 16 lanes * sizeof(f32)
shl r3, ra5, 6
mov rb20, r3

# nbytes = n * sizeof(f32)
shl r1, ra3, 2
mov ra9, r1

# Current x/y byte addresses for this QPU.
add ra10, ra0, r0
add ra11, ra1, r0

# Use one disjoint VPM row per QPU for the output path.
mov ra12, ra4

# If this QPU's initial chunk base is already >= n, do no work.
mov r1, ra9
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

:loop
    # Compute per-lane byte offsets in an accumulator. Do not keep this in
    # regfile A and then add it to an A-regfile base address: one ALU
    # instruction may read at most one value from regfile A and one from
    # regfile B. Using r1 avoids the vc4asm A20 same-regfile read conflict.
    shl r1, elem_num, 2

    # Issue two independent direct-memory TMU0 request vectors before receiving.
    # Each lane requests:
    #   first FIFO entry:  x_current + lane*4
    #   second FIFO entry: y_current + lane*4
    add t0s, ra10, r1
    add t0s, ra11, r1
    nop

    # Receive x first. The receive FIFO preserves request order.
    ldtmu0

    # Use x from r4 while signalling the receive for y. y is available in r4
    # for the following instruction.
    fmul r2, r4, ra2; ldtmu0

    # Complete canonical SAXPY arithmetic: y = alpha * x + y.
    fadd r2, r2, r4

    # Stage the result into this QPU's VPM row.
    mov r3, vpm_setup(1, 1, h32(0))
    add vw_setup, r3, ra12
    mov vpm, r2
    read vw_wait

    # DMA the VPM row to current y memory through VDW.
    shl r1, ra12, 7
    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
    add vw_setup, r2, r1
    mov vw_addr, ra11
    read vw_wait

    # Advance this QPU's worklist by num_qpus vectors.
    add ra8, ra8, rb20
    add ra10, ra10, rb20
    add ra11, ra11, rb20

    # Loop while the next chunk base is still < nbytes.
    mov r1, ra9
    sub.setf r1, ra8, r1
    brr.anyc -, :loop
    nop
    nop
    nop

:end
# End program. The final thread-end instruction and two delay slots avoid
# uniform/VPM/VDW access and physical register address 14.
thrend
nop
nop
