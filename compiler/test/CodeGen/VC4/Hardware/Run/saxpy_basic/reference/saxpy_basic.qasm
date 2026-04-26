.include "../share/vc4inc/vc4.qinc"

# saxpy_basic reference kernel.
#
# This is the basic SAXPY golden built on the VDR/VPM/VDW read/write path:
#   - read one 16-lane f32 vector from x through VDR -> VPM -> QPU regs
#   - read one 16-lane f32 vector from y through VDR -> VPM -> QPU regs
#   - compute y = alpha * x + y with canonical fmul/fadd
#   - write the result through VPM -> VDW back to y
#
# Physical uniform stream order for one launched QPU:
#   ra0 = x base address
#   ra1 = y base address
#   ra2 = alpha (one scalar f32 word; uniform reads are lane-broadcast)
#   ra3 = n element count (consumed to match ABI; launcher restricts n)
#   ra4 = qpu_id
#   ra5 = num_qpus
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif

# Realize two VPM rows per QPU.
#   row 0 -> physical row (2 * qpu_id)
#   row 1 -> physical row (2 * qpu_id + 1)
shl r3, ra4, 1
mov ra6, r3
add ra7, r3, 1

# Each QPU handles one 16-lane f32 vector. Addresses are byte-addressed.
#   byte_offset = qpu_id * 16 lanes * 4 bytes
shl r0, ra4, 6
add ra8, ra0, r0
add ra9, ra1, r0

# DMA x into the physical VPM row for abstract row 0.
mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
shl r1, ra6, 4
add vr_setup, r2, r1
mov vr_addr, ra8
mov -, vr_wait

# DMA y into the physical VPM row for abstract row 1.
mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
shl r1, ra7, 4
add vr_setup, r2, r1
mov vr_addr, ra9
mov -, vr_wait

# Read the staged vectors back from VPM into QPU registers.
mov r2, vpm_setup(1, 1, h32(0))

add vr_setup, r2, ra6
mov r0, vpm
mov -, vw_wait

add vr_setup, r2, ra7
mov r1, vpm
mov -, vw_wait

# Canonical SAXPY arithmetic: y = alpha * x + y.
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
mov vw_addr, ra9
mov -, vw_wait

thrend
mov interrupt, 1
nop
nop
