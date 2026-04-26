.include "../share/vc4inc/vc4.qinc"

# saxpy_tmu_overlap reference kernel.
#
# This is the latency-hiding TMU-backed counterpart to saxpy_basic:
#   - issue independent direct TMU0 memory requests for x and y before the first receive
#   - receive x/y in FIFO order through ldtmu0/r4
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

# Issue two independent direct-memory TMU0 request vectors before receiving.
# Each lane requests:
#   first FIFO entry:  x_base + qpu_id*64 + lane*4
#   second FIFO entry: y_base + qpu_id*64 + lane*4
#
# This is the point of this golden: the final-stage schedule exposes the
# latency-hiding idiom that upstream lowering/scheduling should aim to produce.
add t0s, ra6, r1
add t0s, ra7, r1
nop

# Receive x first. The TMU receive FIFO preserves request order.
ldtmu0

# Use x from r4 while signalling the receive for y. y is available in r4 for
# the following instruction.
fmul r2, r4, ra2; ldtmu0

# Complete canonical SAXPY arithmetic: y = alpha * x + y.
fadd r2, r2, r4

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
