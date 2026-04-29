.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif

shl r0, ra6, 6
mov ra8, r0
shl r0, ra7, 6
mov rb20, r0
shl r0, ra2, 2
mov ra9, r0
mov ra12, ra6

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
mov r1, ra8
add ra11, ra1, r1

shl r1, elem_num, 2
add r2, ra8, r1

sub r3, r2, 4
max r3, r3, 0

mov r1, ra9
sub r1, r1, 4
add r0, r2, 4
min r0, r0, r1

add t0s, ra0, r3
nop
nop
ldtmu0
mov r1, r4

add t0s, ra0, r2
nop
nop
ldtmu0
mov r3, r4

add t0s, ra0, r0
nop
nop
ldtmu0

fmul r0, r1, ra3
fmul r2, r3, ra4
fadd r0, r0, r2
fmul r2, r4, ra5
fadd r2, r0, r2

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

add ra8, ra8, rb20
mov r1, ra9
sub.setf r1, ra8, r1
brr.anyc -, :loop
nop
nop
nop

:end
nop
thrend
nop
nop

