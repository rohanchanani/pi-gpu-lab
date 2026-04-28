.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

mov ra8, ra3
mov rb20, ra4

:vector_check
mov r0, ra8
shl r0, r0, 4
mov ra9, r0

mov r1, ra2
sub.setf r1, ra9, r1
brr.anync -, :end
nop
nop
nop

mov r1, ra2
sub r3, r1, ra9
mov r1, 16
sub.setf -, r3, r1
brr.anync -, :full_vector
nop
nop
nop

mov ra10, r3
brr -, :have_active_count
nop
nop
nop

:full_vector
mov r1, 16
mov ra10, r1

:have_active_count
mov r1, ra9
add r1, r1, elem_num
shl r1, r1, 2
add t0s, ra0, r1
nop
ldtmu0

mov r2, r4
nop

mov r3, r2 << 15
mov r0, [0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
and r3, r3, r0
add r2, r2, r3
nop

mov r3, r2 << 14
mov r0, [0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
and r3, r3, r0
add r2, r2, r3
nop

mov r3, r2 << 12
mov r0, [0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
and r3, r3, r0
add r2, r2, r3
nop

mov r3, r2 << 8
mov r0, [0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1]
and r3, r3, r0
add r2, r2, r3

read mutex_acq

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra8
mov vpm, r2
read vw_wait

mov r1, ra10
shl r1, r1, 8
shl r1, r1, 8
mov r3, 0x80804000
add r2, r3, r1
shl r1, ra8, 7
add vw_setup, r2, r1

shl r1, ra9, 2
add r1, ra1, r1
mov vw_addr, r1
read vw_wait

mov mutex_rel, 0

add ra8, ra8, rb20
brr -, :vector_check
nop
nop
nop

:end
nop; thrend
nop
nop

