.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif

shl r0, ra3, 4
mov ra8, r0

mov r1, ra2
sub.setf r1, ra8, r1
brr.anync -, :partial_ready_zero
nop
nop
nop

mov r1, ra2
sub r3, r1, ra8
mov r1, 16
sub.setf -, r3, r1
brr.anync -, :partial_count_full
nop
nop
nop

mov ra10, r3
brr -, :partial_count_ready
nop
nop
nop

:partial_count_full
mov r1, 16
mov ra10, r1

:partial_count_ready
mov r2, 0
mov ra11, 0

:partial_loop_check
mov r1, ra10
sub.setf r1, ra11, r1
brr.anync -, :partial_write
nop
nop
nop

mov r1, ra8
add r1, r1, ra11
shl r1, r1, 2
add t0s, ra0, r1
nop
ldtmu0
fadd r2, r2, r4

add ra11, ra11, 1
brr -, :partial_loop_check
nop
nop
nop

:partial_ready_zero
mov r2, 0

:partial_write
read mutex_acq
mov r1, ra5
add r1, r1, ra3
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov mutex_rel, 0

sub.setf -, ra4, 1
brr.allz -, :barrier_partial_done
nop
nop
nop
sub.setf -, ra3, 0
brr.allz -, :barrier_partial_leader
nop
nop
nop
:barrier_partial_nonleader
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_partial_done
nop
nop
nop
:barrier_partial_leader
mov r3, ra4
sub r3, r3, 1
:barrier_partial_arrive_loop
sub.setf -, r3, 0
brr.allz -, :barrier_partial_release_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_partial_arrive_loop
nop
nop
nop
:barrier_partial_release_init
mov r3, ra4
sub r3, r3, 1
:barrier_partial_release_loop
sub.setf -, r3, 0
brr.allz -, :barrier_partial_depart_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_partial_release_loop
nop
nop
nop
:barrier_partial_depart_init
mov r3, ra4
sub r3, r3, 1
:barrier_partial_depart_loop
sub.setf -, r3, 0
brr.allz -, :barrier_partial_reset_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_partial_depart_loop
nop
nop
nop
:barrier_partial_reset_init
mov r3, ra4
sub r3, r3, 1
:barrier_partial_reset_loop
sub.setf -, r3, 0
brr.allz -, :barrier_partial_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_partial_reset_loop
nop
nop
nop

:barrier_partial_done
sub.setf -, ra3, 0
brr.allz -, :leader_reduce
nop
nop
nop
brr -, :barrier_final_start
nop
nop
nop

:leader_reduce
mov r2, 0
mov ra11, 0

:leader_loop_check
mov r1, ra4
sub.setf r1, ra11, r1
brr.anync -, :leader_store
nop
nop
nop

mov r1, ra5
add r1, r1, ra11
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, r1
nop
nop
nop
mov r3, vpm
mov mutex_rel, 0
fadd r2, r2, r3

add ra11, ra11, 1
brr -, :leader_loop_check
nop
nop
nop

:leader_store
read mutex_acq
mov r1, 63
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r1, 63
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra1
read vw_wait
mov mutex_rel, 0

:barrier_final_start
sub.setf -, ra4, 1
brr.allz -, :end
nop
nop
nop
sub.setf -, ra3, 0
brr.allz -, :barrier_final_leader
nop
nop
nop
:barrier_final_nonleader
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :end
nop
nop
nop
:barrier_final_leader
mov r3, ra4
sub r3, r3, 1
:barrier_final_arrive_loop
sub.setf -, r3, 0
brr.allz -, :barrier_final_release_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_final_arrive_loop
nop
nop
nop
:barrier_final_release_init
mov r3, ra4
sub r3, r3, 1
:barrier_final_release_loop
sub.setf -, r3, 0
brr.allz -, :barrier_final_depart_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_final_release_loop
nop
nop
nop
:barrier_final_depart_init
mov r3, ra4
sub r3, r3, 1
:barrier_final_depart_loop
sub.setf -, r3, 0
brr.allz -, :barrier_final_reset_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_final_depart_loop
nop
nop
nop
:barrier_final_reset_init
mov r3, ra4
sub r3, r3, 1
:barrier_final_reset_loop
sub.setf -, r3, 0
brr.allz -, :end
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_final_reset_loop
nop
nop
nop

:end
nop; thrend
nop
nop

