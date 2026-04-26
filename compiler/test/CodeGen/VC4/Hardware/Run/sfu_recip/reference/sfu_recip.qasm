.include "../share/vc4inc/vc4.qinc"

# SFU reciprocal reference kernel.
#
# Semantic operation for one 16-lane vector per active QPU:
#   out[i] = recip(x[i])
#
# Physical uniform stream order per QPU:
#   ra0 = x base address
#   ra1 = out base address
#   ra2 = n element count, carried for ABI coverage
#   ra3 = qpu_id
#   ra4 = num_qpus, carried for ABI coverage
#
# This reference intentionally isolates one new datapath: SFU reciprocal.
# The input vector comes from a direct TMU0 memory lookup. The reciprocal is
# issued by writing to the SFU recip register. Two following instructions avoid
# r4, and only the third following instruction consumes the SFU result from r4.
# Output is staged through VPM and written back through VDW.

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

# One 16-lane vector per QPU:
#   base byte offset = qpu_id * 16 lanes * sizeof(f32)
shl r0, ra3, 6
add ra5, ra0, r0
add ra6, ra1, r0

# VPM row used by this QPU for the output path.
mov ra7, ra3

# Lane byte offsets for direct TMU memory lookup addresses.
shl r1, elem_num, 2

# Issue one direct-memory TMU0 request vector for x.
add t0s, ra5, r1

# Prepare the VPM write setup while the TMU request is in flight.
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra7
nop

# Receive x into r4, copy it to r0, and issue the SFU reciprocal.
ldtmu0
mov r0, r4
mov recip, r0

# SFU hazard window. Do not read or write r4 in the two instructions after the
# SFU write. The result is available in r4 on the third following instruction.
nop
nop
mov vpm, r4
read vw_wait

# DMA the VPM row to out memory through VDW.
shl r1, ra7, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra6
read vw_wait

# End program. The final thread-end instruction and two delay slots avoid
# uniform/VPM/VDW access and physical register address 14.
thrend
nop
nop
