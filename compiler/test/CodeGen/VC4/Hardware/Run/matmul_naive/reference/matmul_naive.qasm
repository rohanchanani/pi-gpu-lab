.include "../share/vc4inc/vc4.qinc"

# matmul_naive reference kernel.
#
# Computes row-major C[M,N] = A[M,K] * B[K,N] for f32 matrices.
#
# CUDA-like mapping used by this test:
#   - one QPU user-program request is one logical 16-lane warp;
#   - ELEMENT_NUMBER is the lane id / column within a 16-column tile;
#   - qpu_id and num_qpus are logical launch builtins carried as uniforms;
#   - each logical warp owns rows qpu_id, qpu_id + num_qpus, ...;
#   - for each row, lanes compute one 16-column output tile at a time.
#
# This is intentionally naive: no VPM shared-memory tiling and no barrier.
# It exercises TMU direct global loads, floating-point multiply/add, tail-safe
# VDW stores, and grid-stride row distribution across all active QPUs.
#
# Physical uniform stream order for one launched QPU:
#   ra0 = A base address, compact row-major M*K f32
#   ra1 = B base address, compact row-major K*N f32 plus private guard padding
#   ra2 = C base address, compact row-major M*N f32
#   ra3 = M
#   ra4 = N
#   ra5 = K
#   ra6 = qpu_id        (logical warp id for independent-vector mode)
#   ra7 = num_qpus
mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif

# Constants and loop-invariant values kept in B-regfile where useful so later
# arithmetic can read one A-regfile value and one B-regfile value.
mov rb20, ra7        # row stride in rows = num_qpus
mov rb21, ra5        # K
mov rb22, ra4        # N
mov r0, 15
add r0, r0, 1
mov rb23, r0         # 16 columns per vector tile

# Current output row for this logical QPU and private VPM staging row.
mov ra8, ra6         # row
mov ra12, ra6        # VPM staging row, one per logical QPU

:row_check
# if row >= M: done
mov r1, ra3
sub.setf r1, ra8, r1
brr.anync -, :end
nop
nop
nop

# A row base byte address: A + (row*K)*4.
mul24 r0, ra8, rb21
shl r0, r0, 2
add ra10, ra0, r0

# C row base byte address: C + (row*N)*4.
mul24 r0, ra8, rb22
shl r0, r0, 2
add ra11, ra2, r0

# col = 0
mov ra9, 0

:col_check
# if col >= N: advance to next row
mov r1, rb22
sub.setf r1, ra9, r1
brr.anync -, :row_advance
nop
nop
nop

# store_count = min(16, N - col)
mov r1, rb22
sub r3, r1, ra9
sub.setf -, r3, rb23
brr.anync -, :full_col_tile
nop
nop
nop

# Tail column tile: remaining columns are in r3, 1..15.
mov ra13, r3
brr -, :have_store_count
nop
nop
nop

:full_col_tile
mov ra13, rb23

:have_store_count
# acc = 0.0 in every lane.  Integer zero has the same bits as f32 zero.
mov r2, 0

# k = 0
mov ra15, 0

:k_check
# if k >= K: store accumulated vector
mov r1, rb21
sub.setf r1, ra15, r1
brr.anync -, :k_done
nop
nop
nop

# Queue A[row,k] as a direct TMU0 lookup.  The scalar address is written as a
# vector, so every lane receives the same A value.
shl r1, ra15, 2
add t0s, ra10, r1

# Queue B[k,col+lane] as a direct TMU0 lookup.
# The launcher puts guard padding after compact B so inactive tail lanes in the
# last K row remain inside allocated GPU memory.  Those lanes are never stored.
mul24 r0, ra15, rb22
add r0, r0, ra9
add r0, r0, elem_num
shl r0, r0, 2
add t0s, ra1, r0
nop

# Receive A first, then receive B while preserving A in r3.
ldtmu0
mov r3, r4; ldtmu0

# acc += A * B
fmul r0, r3, r4
fadd r2, r2, r0

# ++k
add ra15, ra15, 1
brr -, :k_check
nop
nop
nop

:k_done
# Store acc to C[row, col : col + store_count).  VPM/VDW setup and access are
# kept under the global mutex, matching the conservative compiler/runtime rule
# until the setup-clobber tests prove a narrower rule is safe.
read mutex_acq

# Stage the result vector into this QPU's VPM row.
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra12
mov vpm, r2
read vw_wait

# Build VDW setup with dynamic DEPTH = store_count:
#   0x80804000 = ID(VDW basic) | UNITS(1) | HORIZ(1) | DEPTH(0)
#   DEPTH lives in bits [22:16].
mov r1, ra13
shl r1, r1, 8
shl r1, r1, 8
mov r3, 0x80804000
add r0, r3, r1
shl r1, ra12, 7
add vw_setup, r0, r1

# Destination address = C row base + col*4.
shl r1, ra9, 2
add r1, ra11, r1
mov vw_addr, r1
read vw_wait

mov mutex_rel, 0

# col += 16
add ra9, ra9, rb23
brr -, :col_check
nop
nop
nop

:row_advance
# row += num_qpus
add ra8, ra8, rb20
brr -, :row_check
nop
nop
nop

:end
# End program. The final thread-end instruction and two delay slots avoid
# uniform/VPM/VDW access and physical register address 14.
thrend
nop
nop
