.include "../share/vc4inc/vc4.qinc"

# TMU-backed read-nop-write reference kernel.
#
# This is intentionally the same semantic copy as Hardware/Run/read_nop_write,
# but the input side uses TMU0 direct-address memory lookups instead of VDR DMA
# loads. The output side remains:
#
#   r4 -> VPM -> VDW DMA store -> result memory
#
# Physical uniform stream order for one launched QPU:
#   ra0 = input base address
#   ra1 = result base address
#   ra2 = word count, currently active_qpus * 16; not used by this exact-count kernel
#   ra3 = qpu_id
#   ra4 = num_qpus; not used by this one-vector-per-QPU kernel
#
# Direct TMU memory lookup:
#   - write absolute byte addresses to t0s
#   - only s is written, so no texture-setup uniform is consumed
#   - one vector result is received through ldtmu0 into r4
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

# One 16-lane vector per QPU:
#   base byte offset = qpu_id * 16 lanes * 4 bytes
shl r0, ra3, 6
add ra5, ra0, r0
add ra6, ra1, r0

# VPM row used by this QPU.
mov ra7, ra3

# Issue one direct-memory TMU0 request vector.
# Each lane requests input_base + qpu_id*64 + lane*4.
shl r1, elem_num, 2
add t0s, ra5, r1

# Prepare the VPM write setup while the TMU request is in flight.
mov r2, vpm_setup(1, 1, h32(0))
add vw_setup, r2, ra7
nop

# Receive TMU result into r4, then write the vector unchanged into VPM.
ldtmu0
mov vpm, r4
read vw_wait

# DMA the VPM row to result memory through VDW.
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
