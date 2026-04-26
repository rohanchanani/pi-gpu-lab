.include "../share/vc4inc/vc4.qinc"

# VC4 read_nop_write reference kernel.
#
# Physical uniform stream order per QPU:
#   ra0 = input base address
#   ra1 = result/output base address
#   ra2 = n word count
#   ra3 = qpu_id
#   ra4 = num_qpus
#
# This test intentionally copies exactly one 16-lane u32 vector per active QPU:
#   result[qpu_id * 16 + lane] = input[qpu_id * 16 + lane]
#
# It proves the VDR -> VPM -> QPU -> VPM -> VDW memory path without arithmetic,
# loops, tails, TMU, SFU, or thread switching.

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

# byte_offset = qpu_id * 16 * sizeof(uint32_t)
shl r0, ra3, 6
add ra5, ra0, r0
add ra6, ra1, r0

# Use one disjoint VPM row per active QPU.
mov ra7, ra3

# DMA one 16-word vector from input memory into VPM row qpu_id.
mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
shl r1, ra7, 4
add vr_setup, r2, r1
mov vr_addr, ra5
mov -, vr_wait

# Read the staged vector from VPM. The explicit nops keep the setup-to-read
# latency conservative for this ground-truth test.
mov r2, vpm_setup(1, 1, h32(0))
add vr_setup, r2, ra7
nop
nop
nop
mov r0, vpm
mov -, vw_wait

# Write the unchanged vector back to the same VPM row.
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra7
mov vpm, r0
mov -, vw_wait

# DMA the unchanged vector from VPM row qpu_id back to result memory.
shl r1, ra7, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra6
mov -, vw_wait

thrend
nop
nop
