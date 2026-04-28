.include "../share/vc4inc/vc4.qinc"

# saxpy_full reference kernel.
#
# Tail-safe TMU-backed SAXPY golden:
#   y[i] = alpha * x[i] + y[i] for i in [0, n)
#
# Differences from saxpy_16:
#   - n may be any value, including 0 and non-multiples of 16.
#   - Each QPU still processes vector chunks at:
#       base_element = qpu_id * 16 + k * num_qpus * 16
#   - The launcher pads the private GPU scratch buffers up to a 16-element
#     boundary so tail-lane TMU reads stay in allocated GPU memory.
#   - The kernel programs VDW DEPTH dynamically for the final partial vector,
#     so only the logical tail element count is stored for the final chunk.
#
# Physical uniform stream order for one launched QPU:
#   ra0 = x base address in padded GPU scratch
#   ra1 = y base address/result address in padded GPU scratch
#   ra2 = alpha (one scalar f32 word; uniform reads are lane-broadcast)
#   ra3 = logical n element count, not padded_n
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

# If this QPU's initial chunk base is already >= nbytes, do no work.
mov r1, ra9
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

:loop
    # remaining_bytes = nbytes - byte_offset.  Move one A-reg source through
    # an accumulator so the subtract does not read two regfile-A values.
    mov r1, ra8
    sub r3, ra9, r1

    # store_count = min(16, remaining_bytes / sizeof(f32)).
    # Branch if remaining_bytes >= 64.  The value is scalar-uniform across lanes.
    mov r1, 64
    sub.setf -, r3, r1
    brr.anync -, :full_chunk
    nop
    nop
    nop

    # Tail chunk: remaining_bytes is 4..60, so remaining_words is 1..15.
    shr r1, r3, 2
    mov ra13, r1
    brr -, :have_store_count
    nop
    nop
    nop

:full_chunk
    mov r1, 16
    mov ra13, r1

:have_store_count
    # Compute per-lane byte offsets in an accumulator.  The launcher pads the
    # private GPU scratch buffers to a 16-element boundary, so these TMU reads
    # are in-bounds even for inactive tail lanes.
    shl r1, elem_num, 2

    # Issue two independent direct-memory TMU0 request vectors before receiving.
    # Each active lane requests:
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
    # Build a VDW setup word with dynamic DEPTH = store_count:
    #   0x80804000 = ID(VDW basic) | UNITS(1) | HORIZ(1) | DEPTH(0)
    #   DEPTH is then OR/add-ed into bits [22:16].  Shifts by 16 are expressed
    #   as two shifts by 8 to stay in well-tested small-immediate shift form.
    mov r1, ra13
    shl r1, r1, 8
    shl r1, r1, 8
    mov r3, 0x80804000
    add r2, r3, r1
    shl r1, ra12, 7
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
