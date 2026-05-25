.include "../share/vc4inc/vc4.qinc"

# Read uniforms into registers
mov   ra0, unif #RESOLUTION
mov   ra1, unif #1/RESOLUTION
mov   ra2, unif #MAX_ITER
mov   ra3, unif #NUM_QPU
mov   ra4, unif #QPU_NUM
mov   ra5, unif #ADDRESS

    #-----------------------------------------------------
    # 1) DMA read of A
    #-----------------------------------------------------

mov ra10, ra4 #i = QPU_NUM

:row_loop

shl ra6, ra0, 1 #m = 2*RESOLUTION
mov ra11, 0 # 2*RESOLUTION

shl r1, ra0, 3 #bytes_per_row = 8*RESOLUTION
mul24 ra12, r1, ra10 #Array_idx = i * bytes_per_row

itof r1, ra10 #(float) i
itof r2, -1 #(float) -1
fmul r1, r1, ra1 #i * 1/RESOLUTION
fadd rb9, r1, r2 #y = -1 + i*1/resolution

#rb9 = y, rb8 = x

:column_loop

    mov r1, ra11 #j
    add r1, r1, elem_num #j+= elem_num
    itof r1, r1 #(float) j
    itof r2, -1 #(float) -1
    fmul r1, r1, ra1 #x = j*1/RESOLUTION
    fadd rb8, r1, r2 #x = -1 + j*1/RESOLUTION


    itof r1, 0
    mov rb0, r1
    mov rb1, r1
    mov rb2, r1
    mov rb3, r1

    mov r1, ra2
    mov ra7, r1

    mov rb7, 0

:inner_loop

    itof r1, 2
    fmul rb1, rb1, r1 #v *= 2
    mov r1, rb0
    fmul rb1, rb1, r1 #v *= u
    mov r1, rb9
    fadd rb1, rb1, r1 #v += y

    mov r1, rb8
    mov r2, rb2
    fadd rb0, r2, r1 #u = u^2 + x

    mov r1, rb3
    fsub rb0, rb0, r1 #u -= v^2
    nop
    nop
    mov r1, rb0
    fmul rb2, r1, r1 #u^2 = u*u

    mov r1, rb1
    fmul rb3, r1, r1 #v^2 = v*v

    itof r1, 4
    mov r2, rb2
    mov r3, rb3
    fadd r2, r2, r3

    fsub.setf r3, r1, r2

    mov.ifn rb7, 1

    sub.setf ra7, ra7, 1
    brr.anynz -, :inner_loop
    nop
    nop
    nop

    mov r2, vpm_setup(1, 1, h32(0))
    add vw_setup, ra4, r2

    mov vpm, rb7 #VPM WRITE
    mov -, vw_wait

    shl r1, ra4, 7
    mov r2, vdw_setup_0(1, 16, dma_h32(0,0))
    add vw_setup, r1, r2


    mov r1, ra11
    shl r1, r1, 2
    add r1, ra12, r1
    add vw_addr, ra5, r1
    mov -, vw_wait

    add ra11, ra11, 16
    mov r1, ra6
    sub.setf r1, ra11, r1
    brr.anyc -, :column_loop

    nop
    nop
    nop

    mov r1, ra3
    add ra10, ra10, r1 # i += NUM_QPU
    mov r1, ra6
    sub.setf r1, ra10, r1  #if i < 2*RESOLUTION
    brr.anyc -, :row_loop
    nop
    nop
    nop

# End of kernel
:end
thrend
mov interrupt, 1
nop

