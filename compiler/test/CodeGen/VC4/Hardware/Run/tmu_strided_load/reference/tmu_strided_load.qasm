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

shl r0, ra7, 6
mov ra9, r0

shl r3, ra8, 6
mov rb20, r3

shl r1, ra2, 2
mov ra10, r1

add ra11, ra1, r0

mov ra12, ra7

mov r1, ra10
sub.setf r1, ra9, r1
brr.anync -, :end
nop
nop
nop

:loop
mov r1, ra9
sub r3, ra10, r1

mov r1, 64
sub.setf -, r3, r1
brr.anync -, :full_chunk
nop
nop
nop

shr r1, r3, 2
mov ra13, r1
brr -, :have_store_count
nop
nop
nop

:full_chunk
mov r1, 16
mov ra13, r1

:have_store_count
shr r1, ra9, 2
add r1, r1, elem_num

mul24 r2, r1, ra4
add r2, r2, ra3
shl r2, r2, 2
add t0s, ra0, r2
nop
ldtmu0

fmul r2, r4, ra5
fadd r2, r2, ra6

read mutex_acq

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra12
mov vpm, r2
read vw_wait

mov r1, ra13
shl r1, r1, 8
shl r1, r1, 8
mov r3, 0x80804000
add r2, r3, r1
shl r1, ra12, 7
add vw_setup, r2, r1
mov vw_addr, ra11
read vw_wait

mov mutex_rel, 0

add ra9, ra9, rb20
add ra11, ra11, rb20

mov r1, ra10
sub.setf r1, ra9, r1
brr.anyc -, :loop
nop
nop
nop

:end
nop; thrend
nop
nop

