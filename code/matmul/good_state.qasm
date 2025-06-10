.include "../share/vc4inc/vc4.qinc"

# Read uniforms into registers
mov   ra0, unif #N
mov   ra1, unif #A    
mov   ra2, unif #B
mov   ra3, unif #C
mov   ra4, unif #NUM_QPU
mov   ra5, unif #QPU_NUM

nop 

mov ra10, ra5    #i = QPU_NUM

:row_loop

mov ra11, 0     # j_base = 0

nop

shl r1, ra0, 2  #bytes_per_row = 4*N

nop

mov r2, ra10

nop

mul24 rb10, r1, r2  #A_base_addr = i * bytes_per_row


:column_loop



    mov ra12, 0 #k = 0

    mov rb0, 0  #Result = 0

:block_loop

    mov ra13, 0 #l_base = 0

    nop

    mov r1, ra11

    nop
    mov r2, ra12

    nop
    
    add r1, r1, r2
    shl r2, ra0, 2      #Bytes_per_col = 4 * N
    mul24 rb11, r1, r2  #B_base_addr = (j+k) * Bytes_per_col

    nop
    mov rb7, 0          #Result = 0

:inner_loop

    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0)) #DMA READ OF 1 item
    shl r1, ra5, 8                                   #DMA READ OFFSET
    shl r3, ra12, 4                                  #ROW K
    add r1, r1, r3
    add vr_setup, r2, r1

    mov r2, rb10            #A_base_addr
    shl r3, ra13, 2         #row_pointer = l_base * (4 bytes)
    add r2, r2, r3          #array_pointer = A_base_addr + row_pointer
    add vr_addr, ra1, r2   #Address = A_unif_addr + A_pointer
    mov -, vr_wait

    ##Read 1 row, increment by 1, start at 0,0
    #mov vw_setup, vpm_setup(1, 1, h32(0)) 
#
    #ldi r3, 0xdeadbeef
    #mov vpm, r3
    #mov -, vw_wait
    
    #For VPM read, use VPM Row at index QPU_NUM * 16 + k (For h32 VPM setup, VPM Y is bits 0:5 p. 57)
    #add vr_setup, r2, r3

    mov r2, vpm_setup(1, 1, h32(0)) 
    shl r3, ra5, 4
    add r3, r3, ra12
    add vr_setup, r2, r3
    mov r1, vpm
    mov -, vr_wait

    nop
    mov rb5, r1




    mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0)) #DMA READ OF 1 item
    shl r1, ra5, 8                                  #DMA READ OFFSET
    shl r3, ra12, 4                                  #ROW K
    add r1, r1, r3
    add vr_setup, r1, r2                            


    mov r2, rb11            #B_base_addr
    shl r3, ra13, 2         #col_pointer = l_base * (4 bytes)
    add r2, r2, r3          #array_pointer = B_base_addr + col_pointer
    add vr_addr, ra2, r2    #Address = B_unif_addr + B_pointer
    mov -, vr_wait

    mov r2, vpm_setup(1, 1, h32(0)) 
    shl r3, ra5, 4
    add r3, r3, ra12
    add vr_setup, r2, r3
    mov r1, vpm
    mov -, vr_wait

    nop
    mov rb5, r1

    

#
#    #Read 1 row, increment by 1, start at 0,0
#    mov r2, vpm_setup(1, 1, h32(0)) 
#    
#    #For VPM read, use VPM Row at index QPU_NUM * 16 + k (For h32 VPM setup, VPM Y is bits 0:5 p. 57)
#    shl r3, ra5, 4
#    add r3, r3, ra12
#    add vr_setup, r2, r3
#
#    mov rb4, vpm
#    mov -, vw_wait
#
#    mov r1, rb4
#    mov r2, rb5
#
#    mul24 r1, r1, r2
#    mov r2, rb7
#    add rb7, r1, r2
#
#    add ra13, ra13, 16
#    nop
#    mov r1, ra13
#    nop
#    sub.setf -, ra0, r1
#    brr.anynz -, :inner_loop
#    nop
#    nop
#    nop
#
#    #Read 1 row, increment by 1, start at 0,0
#    mov r2, vpm_setup(1, 1, h32(0)) 
#    
#    #For VPM read, use VPM Row at index QPU_NUM * 16 + k (For h32 VPM setup, VPM Y is bits 0:5 p. 57)
#    shl r3, ra5, 4
#    add r3, r3, ra12
#    add vw_setup, r2, r3
#
#    mov vpm, rb7
#    mov -, vw_wait
#
#    add ra13, ra13, 1
#    mov r1, 16
#    sub.setf -, r1, ra13
#    brr.anynz -, :block_loop
#    nop
#    nop
#    nop
#
#
#    mov r1, vpm_setup(16, 1, h32(0))
#    shl r3, ra5, 4
#    add vr_setup, r1, r3
#
#    mov rb16, vpm
#    mov -, vw_wait
#    mov rb15, vpm
#    mov -, vw_wait
#    mov rb14, vpm
#    mov -, vw_wait
#    mov rb13, vpm
#    mov -, vw_wait
#    mov rb12, vpm
#    mov -, vw_wait
#    mov rb11, vpm
#    mov -, vw_wait
#    mov rb10, vpm
#    mov -, vw_wait
#    mov rb9, vpm
#    mov -, vw_wait
#    mov rb8, vpm
#    mov -, vw_wait
#    mov rb7, vpm
#    mov -, vw_wait
#    mov rb6, vpm
#    mov -, vw_wait
#    mov rb5, vpm
#    mov -, vw_wait
#    mov rb4, vpm
#    mov -, vw_wait
#    mov rb3, vpm
#    mov -, vw_wait
#    mov rb2, vpm
#    mov -, vw_wait
#    mov rb1, vpm
#    mov -, vw_wait
#
#    mov r1, vpm_setup(16, 1, v32(0))
#    shl r3, ra5, 4
#    add vw_setup, r1, r3
#
#    mov vpm, rb16
#    mov -, vw_wait
#    mov vpm, rb15
#    mov -, vw_wait
#    mov vpm, rb14
#    mov -, vw_wait
#    mov vpm, rb13
#    mov -, vw_wait
#    mov vpm, rb12
#    mov -, vw_wait
#    mov vpm, rb11
#    mov -, vw_wait
#    mov vpm, rb10
#    mov -, vw_wait
#    mov vpm, rb9
#    mov -, vw_wait
#    mov vpm, rb8
#    mov -, vw_wait
#    mov vpm, rb7
#    mov -, vw_wait
#    mov vpm, rb6
#    mov -, vw_wait
#    mov vpm, rb5
#    mov -, vw_wait
#    mov vpm, rb4
#    mov -, vw_wait
#    mov vpm, rb3
#    mov -, vw_wait
#    mov vpm, rb2
#    mov -, vw_wait
#    mov vpm, rb1
#    mov -, vw_wait
#
#
#    mov rb0, 0
#    #Read back out the rows
#    mov r1, vpm_setup(16, 1, h32(0))
#    shl r3, ra5, 4
#    add vr_setup, r1, r3
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    ov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
#    mov rb1, vpm
#    mov -, vw_wait
#    mov r1, rb1
#    add rb0, rb0, r1
#
    #Write 1 row, increment by 1, start at 0,0
    mov r2, vpm_setup(1, 1, h32(0)) 
    
    #For VPM write, use VPM Row at index QPU_NUM * 16  (For h32 VPM setup, VPM Y is bits 0:5 p. 57)
    shl r3, ra5, 4
    add vw_setup, r2, r3

    mov vpm, rb5
    mov -, vw_wait

    mov r2, vdw_setup_0(1, 16, dma_h32(0, 0)) #DMA write OF 1 item
    shl r1, ra5, 11                                 #DMA WRITE OFFSET
    add vw_setup, r1, r2


    mov r2, rb11            #C_base_addr
    shl r3, ra13, 2         #row_pointer = j_base * (4 bytes)
    add r2, r2, r3          #array_pointer = C_base_addr + row_pointer
    add vw_addr, ra3, r2    #Address = C_unif_addr + A_pointer
    mov -, vw_wait

    add ra13, ra13, 16
    nop
    mov r1, ra13
    nop
    sub.setf -, ra0, r1
    brr.anynz -, :inner_loop
    nop
    nop
    nop

    add ra12, ra12, 1
    mov r1, 16
    sub.setf -, r1, ra12
    brr.anync -, :block_loop
    nop
    nop
    nop

    add ra11, ra11, 16
    mov r1, ra0
    sub.setf r1, ra11, r1
    brr.anyc -, :column_loop
    
    nop
    nop
    nop

    mov r1, ra4
    add ra10, ra10, r1

    nop


    mov r1, ra0
    sub.setf r1, ra10, r1
    brr.anyc -, :row_loop

    nop
    nop
    nop

# End of kernel
:end
thrend
mov interrupt, 1
nop