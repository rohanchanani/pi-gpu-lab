.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif
mov ra14, unif

mov rb20, 128
mov rb22, ra14
mov ra8, ra7

:row_check
mov r1, ra3
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

shl r0, ra8, 7
mov ra10, r0

mov r0, ra8
sub r0, r0, 1
max r0, r0, 0
shl r0, r0, 7
mov ra11, r0

mov r3, ra8
add r3, r3, 1
mov r1, ra3
sub r1, r1, 1
min r3, r3, r1
shl r3, r3, 7
mov ra12, r3

mov ra9, 0

:col_check
mov r1, ra2
sub.setf r1, ra9, r1
brr.anync -, :row_next
nop
nop
nop

mov r1, ra2
sub r3, r1, ra9
mov r1, 16
sub.setf -, r3, r1
brr.anync -, :full_tile
nop
nop
nop

mov ra13, r3
brr -, :have_store_count
nop
nop
nop

:full_tile
mov ra13, 16

:have_store_count
mov r1, ra9
add r1, r1, elem_num
shl r2, r1, 2
add r2, ra10, r2
add t0s, ra0, r2
nop
nop
ldtmu0
mov ra15, r4

mov r1, ra9
add r1, r1, elem_num
shl r2, r1, 2
add r2, ra11, r2
add t0s, ra0, r2
nop
nop
ldtmu0
mov ra16, r4

mov r1, ra9
add r1, r1, elem_num
shl r2, r1, 2
add r2, ra12, r2
add t0s, ra0, r2
nop
nop
ldtmu0
mov ra17, r4

mov r1, ra9
add r1, r1, elem_num
sub r2, r1, 1
max r2, r2, 0
shl r2, r2, 2
add r2, ra10, r2
add t0s, ra0, r2
nop
nop
ldtmu0
mov ra18, r4

mov r1, ra9
add r1, r1, elem_num
add r2, r1, 1
mov r3, ra2
sub r3, r3, 1
min r2, r2, r3
shl r2, r2, 2
add r2, ra10, r2
add t0s, ra0, r2
nop
nop
ldtmu0

mov r0, ra16
fadd r0, r0, ra17
fadd r0, r0, ra18
fadd r0, r0, r4
fmul r0, r0, ra6

mov r1, ra15
fmul r1, r1, ra5
fadd r2, r1, r0

read mutex_acq

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra7
mov vpm, r2
read vw_wait

mov r1, ra13
shl r1, r1, 8
shl r1, r1, 8
mov r3, 0x80804000
add r2, r3, r1
shl r1, ra7, 7
add vw_setup, r2, r1

shl r1, ra9, 2
add r1, ra10, r1
add r1, ra1, r1
mov vw_addr, r1
read vw_wait

mov mutex_rel, 0

add ra9, ra9, 16
brr -, :col_check
nop
nop
nop

:row_next
add ra8, ra8, rb22
brr -, :row_check
nop
nop
nop

:end
nop
thrend
nop
nop

