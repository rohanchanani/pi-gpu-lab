.include "../share/vc4inc/vc4.qinc"

# saxpy_tmu reference kernel.
#
# This is the TMU-backed counterpart to saxpy_basic:
#   - read one 16-lane f32 vector from x through TMU0 direct memory lookup
#   - read one 16-lane f32 vector from y through TMU0 direct memory lookup
#   - compute y = alpha * x + y with canonical fmul/fadd
#   - write the result through VPM -> VDW back to y
#
# Physical uniform stream order for one launched QPU:
#   ra0 = x base address
#   ra1 = y base address/result address
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

# One 16-lane vector per QPU:
#   base byte offset = qpu_id * 16 lanes * 4 bytes
shl r0, ra4, 6
add ra6, ra0, r0
add ra7, ra1, r0

# VPM row used by this QPU for the output path.
mov ra8, ra4

# Lane byte offsets for direct TMU memory lookup addresses.
shl r1, elem_num, 2

# Issue one direct-memory TMU0 request vector for x.
# Each lane requests x_base + qpu_id*64 + lane*4.
add t0s, ra6, r1
nop
ldtmu0
mov r0, r4

# Issue one direct-memory TMU0 request vector for y.
# Each lane requests y_base + qpu_id*64 + lane*4.
add t0s, ra7, r1
nop
ldtmu0
mov r1, r4

# Canonical SAXPY arithmetic: y = alpha * x + y.
fmul r2, r0, ra2
fadd r2, r2, r1

# Stage the result into this QPU's VPM row.
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra8
mov vpm, r2
read vw_wait

# DMA the VPM row to y memory through VDW.
shl r1, ra8, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra7
read vw_wait

# End program. The final thread-end instruction and two delay slots avoid
# uniform/VPM/VDW access and physical register address 14.
thrend
nop
nop
