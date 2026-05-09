.include "../share/vc4inc/vc4.qinc"

# VC4 memory_output reference kernel.
#
# Uniform stream order per launched QPU:
#   ra0 = output base bus address
#   ra1 = qpu_id
#   ra2 = num_qpus
#
# Each active QPU writes one 16-lane u32 vector to:
#
#   output + qpu_id * 16
#
# The value in every lane of QPU q's row is q.  The host harness checks the
# resulting 12 x 16 words against this semantic oracle.
mov ra0, unif
mov ra1, unif
mov ra2, unif

# Per-QPU output address in bytes: base + qpu_id * 16 lanes * 4 bytes.
shl r0, ra1, 6
add ra3, ra0, r0

# Data vector: every lane gets the scalar qpu_id uniform.
mov r0, ra1

# Write the vector to one VPM row per QPU.
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra1
mov vpm, r0
mov -, vw_wait

# DMA that row to the QPU's output slice.
shl r1, ra1, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra3
mov -, vw_wait

# End program with the required two delay-slot instructions.
thrend
nop
nop
