.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif
mov ra8, unif

mov r0, qpu_num

sub.setf -, r0, ra0
brr.allz -, :role_a
nop
nop
nop

sub.setf -, r0, ra1
brr.allz -, :role_b
nop
nop
nop

brr -, :end
nop
nop
nop

:role_a
srel -, 0
sacq -, 1

mov r2, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra3
mov vpm, r2
read vw_wait
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

srel -, 2

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra3

srel -, 3
sacq -, 4

mov r2, 0xe5000000
and r3, ra0, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

mov vpm, r2
read vw_wait

srel -, 5

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 62
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 62
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra7
read vw_wait
mov mutex_rel, 0

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra3
nop
nop
nop
mov r2, vpm
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 60
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 60
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra5
read vw_wait
mov mutex_rel, 0

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra4
nop
nop
nop
mov r2, vpm
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 61
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 61
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra6
read vw_wait
mov mutex_rel, 0

brr -, :end
nop
nop
nop

:role_b
srel -, 1
sacq -, 0

sacq -, 2

sacq -, 3
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4

srel -, 4

sacq -, 5

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 62
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 62
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra8
read vw_wait
mov mutex_rel, 0

brr -, :end
nop
nop
nop

:end
nop; thrend
nop
nop

