.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif

shl r0, ra2, 2
mov ra8, r0
shl r0, elem_num, 2
mov ra10, r0

mov r1, ra8
shl r2, r1, 6
add r2, r2, ra10
add t0s, ra0, r2
nop
nop
ldtmu0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add r2, ra4, r1
add vw_setup, r3, r2
mov vpm, r4
read vw_wait
mov mutex_rel, 0

add r1, ra8, 1
shl r2, r1, 6
add r2, r2, ra10
add t0s, ra0, r2
nop
nop
ldtmu0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add r2, ra4, r1
add vw_setup, r3, r2
mov vpm, r4
read vw_wait
mov mutex_rel, 0

add r1, ra8, 2
shl r2, r1, 6
add r2, r2, ra10
add t0s, ra0, r2
nop
nop
ldtmu0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add r2, ra4, r1
add vw_setup, r3, r2
mov vpm, r4
read vw_wait
mov mutex_rel, 0

add r1, ra8, 3
shl r2, r1, 6
add r2, r2, ra10
add t0s, ra0, r2
nop
nop
ldtmu0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add r2, ra4, r1
add vw_setup, r3, r2
mov vpm, r4
read vw_wait
mov mutex_rel, 0

sub.setf -, ra2, 0
brr.allz -, :barrier_leader
nop
nop
nop

srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_done
nop
nop
nop

:barrier_leader
mov r3, ra3
sub r3, r3, 1

:barrier_arrive_loop
sub.setf -, r3, 0
brr.allz -, :barrier_release_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_arrive_loop
nop
nop
nop

:barrier_release_init
mov r3, ra3
sub r3, r3, 1

:barrier_release_loop
sub.setf -, r3, 0
brr.allz -, :barrier_depart_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_release_loop
nop
nop
nop

:barrier_depart_init
mov r3, ra3
sub r3, r3, 1

:barrier_depart_loop
sub.setf -, r3, 0
brr.allz -, :barrier_reset_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_depart_loop
nop
nop
nop

:barrier_reset_init
mov r3, ra3
sub r3, r3, 1

:barrier_reset_loop
sub.setf -, r3, 0
brr.allz -, :barrier_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_reset_loop
nop
nop
nop

:barrier_done
mov r1, ra8
read mutex_acq
mov r3, vpm_setup(1, 1, v32(0, 0))
add vr_setup, r3, r1
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r0, 63
add vw_setup, r3, r0
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r0, 63
shl r0, r0, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r0
shl r0, r1, 6
add r0, ra1, r0
mov vw_addr, r0
read vw_wait
mov mutex_rel, 0

add r1, ra8, 1
read mutex_acq
mov r3, vpm_setup(1, 1, v32(0, 0))
add vr_setup, r3, r1
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r0, 63
add vw_setup, r3, r0
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r0, 63
shl r0, r0, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r0
shl r0, r1, 6
add r0, ra1, r0
mov vw_addr, r0
read vw_wait
mov mutex_rel, 0

add r1, ra8, 2
read mutex_acq
mov r3, vpm_setup(1, 1, v32(0, 0))
add vr_setup, r3, r1
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r0, 63
add vw_setup, r3, r0
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r0, 63
shl r0, r0, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r0
shl r0, r1, 6
add r0, ra1, r0
mov vw_addr, r0
read vw_wait
mov mutex_rel, 0

add r1, ra8, 3
read mutex_acq
mov r3, vpm_setup(1, 1, v32(0, 0))
add vr_setup, r3, r1
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r0, 63
add vw_setup, r3, r0
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r0, 63
shl r0, r0, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r0
shl r0, r1, 6
add r0, ra1, r0
mov vw_addr, r0
read vw_wait
mov mutex_rel, 0

nop
thrend
nop
nop

