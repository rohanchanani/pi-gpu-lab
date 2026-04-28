.include "../share/vc4inc/vc4.qinc"

# matmul_blocked reference kernel.
#
# CUDA-like blocked f32 matmul:
#   C[M,N] = A[M,K] * B[K,N]
#
# One resident cooperative block computes one 12-row x 16-column C tile.
#   - one QPU request is one logical 16-lane warp;
#   - logical_warp_id selects the row inside the block tile;
#   - ELEMENT_NUMBER selects the column lane inside the 16-column tile;
#   - VPM rows vpm_base_row .. vpm_base_row+11 hold one shared B K-tile;
#   - semaphores 0..3 implement the four-semaphore reusable block barrier;
#   - VPM/VDW setup, VPM reads/writes, and VDW stores are mutex protected.
#
# Physical uniform stream order for one logical warp request:
#   ra0  = A scratch base, row-major with stride padded_k_capacity
#   ra1  = B scratch base, row-major with stride padded_n_capacity
#   ra2  = C scratch base, row-major with stride padded_n_capacity
#   ra3  = logical M
#   ra4  = logical N
#   ra5  = logical K, documented ABI only; padding controls the loop bound
#   ra6  = padded_k_capacity / A row stride
#   ra7  = padded_n_capacity / B and C row stride
#   ra8  = tile_row_base
#   ra9  = tile_col_base
#   ra10 = logical_warp_id in the resident block, 0..11
#   ra11 = warps_per_block / K tile rows, normally 12
#   ra12 = padded K loop bound for this launch, multiple of warps_per_block
#   ra13 = vpm_base_row, normally 0
#
# The launcher pads A and B scratch memory with zeroes.  The qasm therefore
# computes over padded K rows and padded N columns, but stores only logical
# columns using dynamic VDW DEPTH.  Rows beyond M are written only to the
# private padded C scratch and are never copied back to the semantic host C.

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif
mov ra8, unif
mov ra9, unif
mov ra10, unif
mov ra11, unif
mov ra12, unif
mov ra13, unif

# Keep stride/shape values in the B regfile so multiply/address instructions
# can read one A-regfile source and one B-regfile source.
mov rb20, ra6        # A row stride = padded_k_capacity
mov rb21, ra7        # B/C row stride = padded_n_capacity
mov rb22, ra11       # warps_per_block, also K tile size
mov rb23, ra12       # padded K loop bound for this launch
mov rb24, ra13       # VPM base row for shared B tile

# row = tile_row_base + logical_warp_id.
mov r0, ra8
add r0, r0, ra10
mov ra15, r0

# store_count = min(16, N - tile_col_base).  Host launches only tile_col < N.
mov r0, 15
add r0, r0, 1
mov ra13, r0         # default 16 columns
mov r1, ra4
sub r3, r1, ra9      # remaining columns in this tile
sub.setf -, r3, ra13
brr.anync -, :store_count_ready
nop
nop
nop
mov ra13, r3         # tail columns, 1..15
:store_count_ready

# acc = 0.0 in all lanes.
mov r2, 0

# kbase = 0.
mov ra16, 0

:k_tile_check
# if kbase >= padded_k_limit: store accumulated C vector.
mov r1, rb23
sub.setf r1, ra16, r1
brr.anync -, :store_c_vector
nop
nop
nop

# Each logical warp loads one B row for this K tile into shared VPM:
#   sharedB[logical_warp_id, lane] = B[kbase + logical_warp_id,
#                                      tile_col_base + lane]
mov r0, ra16
add r0, r0, ra10
mul24 r0, r0, rb21
add r0, r0, ra9
add r0, r0, elem_num
shl r0, r0, 2
add t0s, ra1, r0
nop
ldtmu0
mov r3, r4

read mutex_acq
mov r1, rb24
add r1, r1, ra10
mov r0, vpm_setup(1, 1, h32(0))
add vw_setup, r0, r1
mov vpm, r3
read vw_wait
mov mutex_rel, 0

# Barrier #1: all B rows for this K tile are in VPM before any warp reads them.
sub.setf -, ra10, 0
brr.allz -, :barrier_load_leader
nop
nop
nop
:barrier_load_nonleader
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_load_done
nop
nop
nop
:barrier_load_leader
mov r3, rb22
sub r3, r3, 1
:barrier_load_arrive_loop
sub.setf -, r3, 0
brr.allz -, :barrier_load_release_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_load_arrive_loop
nop
nop
nop
:barrier_load_release_init
mov r3, rb22
sub r3, r3, 1
:barrier_load_release_loop
sub.setf -, r3, 0
brr.allz -, :barrier_load_depart_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_load_release_loop
nop
nop
nop
:barrier_load_depart_init
mov r3, rb22
sub r3, r3, 1
:barrier_load_depart_loop
sub.setf -, r3, 0
brr.allz -, :barrier_load_reset_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_load_depart_loop
nop
nop
nop
:barrier_load_reset_init
mov r3, rb22
sub r3, r3, 1
:barrier_load_reset_loop
sub.setf -, r3, 0
brr.allz -, :barrier_load_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_load_reset_loop
nop
nop
nop
:barrier_load_done

# p = 0 .. warps_per_block-1.  For each p, load A[row,kbase+p] as a scalar
# vector and read shared B[p,lane] from VPM, then accumulate A*B.
mov ra17, 0
:p_loop_check
mov r1, rb22
sub.setf r1, ra17, r1
brr.anync -, :p_loop_done
nop
nop
nop

# Queue A[row, kbase+p] as direct TMU0 lookup.  The address vector is the
# same in every lane, so every lane receives the scalar A value.
mul24 r0, ra15, rb20
add r0, r0, ra16
add r0, r0, ra17
shl r0, r0, 2
add t0s, ra0, r0

# Read shared B[p,lane] from VPM while the A lookup is in flight.
read mutex_acq
mov r1, rb24
add r1, r1, ra17
mov r0, vpm_setup(1, 1, h32(0))
add vr_setup, r0, r1
nop
nop
nop
mov r3, vpm
mov mutex_rel, 0

ldtmu0
fmul r0, r4, r3
fadd r2, r2, r0

add ra17, ra17, 1
brr -, :p_loop_check
nop
nop
nop

:p_loop_done
# Barrier #2: every warp has finished reading the shared B tile before any
# warp overwrites those VPM rows for the next K tile.
sub.setf -, ra10, 0
brr.allz -, :barrier_read_leader
nop
nop
nop
:barrier_read_nonleader
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_read_done
nop
nop
nop
:barrier_read_leader
mov r3, rb22
sub r3, r3, 1
:barrier_read_arrive_loop
sub.setf -, r3, 0
brr.allz -, :barrier_read_release_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_read_arrive_loop
nop
nop
nop
:barrier_read_release_init
mov r3, rb22
sub r3, r3, 1
:barrier_read_release_loop
sub.setf -, r3, 0
brr.allz -, :barrier_read_depart_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_read_release_loop
nop
nop
nop
:barrier_read_depart_init
mov r3, rb22
sub r3, r3, 1
:barrier_read_depart_loop
sub.setf -, r3, 0
brr.allz -, :barrier_read_reset_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_read_depart_loop
nop
nop
nop
:barrier_read_reset_init
mov r3, rb22
sub r3, r3, 1
:barrier_read_reset_loop
sub.setf -, r3, 0
brr.allz -, :barrier_read_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_read_reset_loop
nop
nop
nop
:barrier_read_done

# kbase += warps_per_block.
add ra16, ra16, rb22
brr -, :k_tile_check
nop
nop
nop

:store_c_vector
# Store acc to C[row, tile_col : tile_col + store_count).  The result vector is
# staged through VPM row 63 and DMA-stored with dynamic DEPTH for N tails.
read mutex_acq

mov r1, 63
mov r0, vpm_setup(1, 1, h32(0))
add vw_setup, r0, r1
mov vpm, r2
read vw_wait

# Build VDW setup word with dynamic DEPTH = store_count:
#   0x80804000 = ID(VDW basic) | UNITS(1) | HORIZ(1) | DEPTH(0)
#   DEPTH lives in bits [22:16].
mov r1, ra13
shl r1, r1, 8
shl r1, r1, 8
mov r3, 0x80804000
add r0, r3, r1
mov r1, 63
shl r1, r1, 7
add vw_setup, r0, r1

# Destination address = C + (row * padded_n_capacity + tile_col_base) * 4.
mul24 r0, ra15, rb21
add r0, r0, ra9
shl r0, r0, 2
add r0, ra2, r0
mov vw_addr, r0
read vw_wait

mov mutex_rel, 0

:end
# End program.  The final thread-end instruction and two delay slots avoid
# uniforms, VPM/VDR/VDW, and physical register-file address 14.
thrend
nop
nop
