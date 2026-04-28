.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

shl r0, ra3, 6
mov ra8, r0

shl r3, ra4, 6
mov rb20, r3

shl r1, ra1, 2
mov ra9, r1

add ra10, ra0, r0

mov ra12, ra3

mov r1, ra9
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

:loop
mov r1, ra8
sub r3, ra9, r1

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
shr r1, ra8, 2
add r1, r1, elem_num

mov r2, 0x51000000
mov r3, ra2
shl r3, r3, 8
shl r3, r3, 8
or r2, r2, r3
or r2, r2, r1

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
mov vw_addr, ra10
read vw_wait

mov mutex_rel, 0

add ra8, ra8, rb20
add ra10, ra10, rb20

mov r1, ra9
sub.setf r1, ra8, r1
brr.anyc -, :loop
nop
nop
nop

:end
nop; thrend
nop
nop

