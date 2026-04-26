.include "../share/vc4inc/vc4.qinc"

# qpu_num_register reference kernel.
#
# Physical uniform stream order for one scheduler request:
#   ra0 = output scratch row bus address
#
# The kernel reads the hardware qpu_num / QPU_NUMBER register, stages the
# observed value through a per-QPU VPM row, and stores one 32-bit word with VDW.

mov ra0, unif

# r0 = hardware QPU number.  vc4asm maps qpu_num to the B-regfile QPU_NUMBER
# register at QPU register-map address 38.
mov r0, qpu_num

# Use VPM row qpu_num so concurrently running QPUs do not fight over row 0.
mov r1, vpm_setup(1, 1, h32(0))
add vw_setup, r1, r0
mov vpm, r0
read vw_wait

# Program a one-word VDW store from the selected VPM row to this request's
# output scratch row.  The VDW setup takes the VPM base row from element 0.
mov r1, vdw_setup_1(0)
mov vw_setup, r1
mov r1, vdw_setup_0(1, 1, dma_h32(0, 0))
shl r2, r0, 7
add vw_setup, r1, r2
mov vw_addr, ra0
read vw_wait

# End program. The final thread-end instruction and two delay slots avoid
# uniform/VPM/VDW access and physical register address 14.
thrend
nop
nop
