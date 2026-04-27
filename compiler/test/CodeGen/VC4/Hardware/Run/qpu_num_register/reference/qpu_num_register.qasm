.include "../share/vc4inc/vc4.qinc"

# qpu_num_register targeted reference kernel.
#
# Uniform stream for one scheduler request:
#   unif[0] = output slot bus address
#
# The launcher reserves every physical QPU except the target before queueing
# this program.  Therefore the value written back should equal the target slot.
# The VPM/VDW path is still protected by the global QPU mutex so this shader is
# safe even if it is later reused in a concurrent launch.

# Preserve this request's output slot address.
mov ra0, unif

# r0 = physical QPU number.  vc4asm maps qpu_num to the B-regfile QPU_NUMBER
# register at QPU register-map address 38.
mov r0, qpu_num

# Independent cross-check channel: latch this physical QPU's HOST_INT bit.
# HOST_INT must be written with a non-zero value.
mov interrupt, 1

# Serialize the VPM/VDW setup and DMA store.  The single-QPU reservation should
# already avoid races, but this keeps the shader robust if reused elsewhere.
read mutex_acq

# Stage one 32-bit word through VPM row 0, column 0.
mov vw_setup, vpm_setup(1, 1, h32(0))
mov vpm, r0
read vw_wait

# Store one 32-bit word from VPM row 0, column 0 to this target's output slot.
mov vw_setup, vdw_setup_1(0)
mov vw_setup, vdw_setup_0(1, 1, dma_h32(0, 0))
mov vw_addr, ra0
read vw_wait

# Release global QPU mutex.
mov mutex_rel, 1

# End program.  The final three instructions avoid uniforms/VPM/VDW accesses
# and avoid physical register-file address 14.
thrend
nop
nop
