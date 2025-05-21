.include "../share/vc4inc/vc4.qinc"

# Read uniforms into registers
mov   ra0, unif #RESOLUTION    
mov   ra1, unif #1/RESOLUTION
mov   ra2, unif #MAX_ITER
mov   ra3, unif #NUM_QPU
mov   ra4, unif #QPU_NUM
mov   ra5, unif #ADDRESS

mov ra10, ra4           #i = QPU_NUM
mov r1, ra0
shl ra6, r1, 1         #width,height = 2*RESOLUTION

:row_loop               #We'll use a nested 2D loop like the CPU example

mov ra11, 0             #j_base = 0

shl r1, ra0, 3          #bytes_per_row = 2*RESOLUTION*sizeof(uint32_t) = 8*RESOLUTION

mov r2, ra10
mul24 ra12, r1, r2      #row_base_address = i * bytes_per_row 

itof r1, ra10           #(float) i

#TODO: Use r1 (i) to determine the float y value you'll compute (see CPU example)

#rb9 = y, rb8 = x

:column_loop

    #TODO: Use j_base and the elem_num register to determine the float x value you'll compute
    #Recall that we have a 16-wide vector, and ra11 (j_base) holds the leftmost col index in the row.
    #Use j_base and the elem_num register to figure out the exact j in the output array, then follow the 
    #CPU example to determine the corresponding float x example
	

    #Initialize u, v, u2, v2 to 0 (all floats)
    itof r1, 0
    mov rb0, r1     #u
    mov rb1, r1     #v
    mov rb2, r1     #u2
    mov rb3, r1     #v2
      
    mov ra7, ra2    #Convert max iters into a function
    
    mov rb7, 0      #THIS WILL EVENTUALLY BE OUR OUTPUT VALUE

:inner_loop

    #TODO: MODEL THE CPU EXAMPLE TO UPDATE U,V, U2, V2 correctly
    

    #TODO: CHECK FOR DIVERGENCE (u^2+v^2>4), AND USE CONDITION CODE
   
    #ADD SOME INSTRUCTION THAT SET FLAGS IF DIVERGED

    #TODO: ADD THE CONDITION CODE THAT MATCHES YOUR INSTRUCTION
    mov.<condition for diverged> rb7, 1

    #IF ALL LANES HAVE DIVERGED, WE CAN ESCAPE THE LOOP
    brr.all<condition for diverged> -, :exit
    nop
    nop
    nop

    #UPDATE OUR MAX ITERS COUNTER
    sub.setf ra7, ra7, 1
    brr.anynz -, :inner_loop
    nop
    nop
    nop

    :exit

    #WRITE OUR 16-wide vector out to VPM, in the row for our QPU.

    mov r2, vpm_setup(1, 1, h32(0))     #WRITING 1 ROW, STARTING FROM UPPER LEFT
    add vw_setup, ra4, r2               #We can do math on vpm_setup, to get nth row just add n (docs p. 57-59)
                                        #In this case, we write to the row corresponding to our QPU_NUM, to avoid conflict

    mov vpm, rb7                        #Write the row to VPM
    mov -, vw_wait

    #DMA THE 16-wide row from VPM to PHYSICAL MEMORY

    #For the DMA registers, the bits corresponding to the row aren't just the lowest
    #bits, so we have to shift accordingly. See docs p. 57-59

    shl r1, ra4, 7                       
    mov r2, vdw_setup_0(1, 16, dma_h32(0,0))
    add vw_setup, r1, r2
    

    #j_base (index of leftmost column in our 16-wide row is in ra11)
    mov r1, ra11

    #Multiply by 4 because our values are 4-byte uint32_t        
    shl r1, r1, 2

    #Add row_base_address
    add r1, ra12, r1 

    #Kick of the DMA write
    add vw_addr, ra5, r1
    mov -, vw_wait

    #Update our j_base value and check the column loop condition
    
    #We're adding 16 because we just finished computing a 16-wide SIMD vector
    add ra11, ra11, 16

    # Is j_base < width? 
    mov r1, ra6
    sub.setf r1, ra11, r1
    brr.anyc -, :column_loop

    nop
    nop
    nop

    #Update our i value and check the row loop condition

    #We're adding NUM_QPU because the QPUs are computing NUM_QPU rows in SPMD parallel fashion
    mov r1, ra3
    add ra10, ra10, r1         # i += NUM_QPU

    #Is i < height (2*RESOLUTION)? 
    mov r1, ra6
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

