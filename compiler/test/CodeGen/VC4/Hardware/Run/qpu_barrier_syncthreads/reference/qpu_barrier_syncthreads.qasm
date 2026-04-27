.include "../share/vc4inc/vc4.qinc"

# qpu_barrier_syncthreads reference kernel.
#
# One generated-style QPU program covers all requested barrier modes.
# Physical uniform stream per request:
#   ra0  mode
#   ra1  run_id
#   ra2  block_id
#   ra3  logical_warp_id
#   ra4  warps_per_block
#   ra5  iterations
#   ra6  vpm_base_row
#   ra7  vpm_rows_per_block
#   ra8  arrive_sem      (documented ABI; semaphore IDs are immediate in VC4)
#   ra9  release_sem     (documented ABI; not dynamically encoded)
#   ra10 depart_sem      (documented ABI; overwritten after uniforms)
#   ra11 reset_sem       (documented ABI; overwritten after uniforms)
#   ra12 result_warp_ptr
#   ra13 run_result_ptr  (reserved for future scalar first-mismatch stores)
#
# The QPU semaphore instruction encodes the semaphore number as a 4-bit
# immediate field, so this qasm dispatches statically by block_id:
# block 0 uses semaphores 0..3 and block 1 uses semaphores 4..7.
#
# The VPM access and VDW result-store sequences are protected by the
# global QPU mutex.  This test is about reusable semaphore barriers, not
# VPM setup-state sharing.
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

# Scratch/state registers after uniform load:
#   ra8  iteration counter
#   ra9  all_seen_mask accumulator
#   ra10 always_seen_mask accumulator
#   ra11 vector-mismatch count
mov r0, qpu_num
mov ra8, 0
mov ra9, 0
mov ra10, 0xffff
mov ra11, 0

:iter_loop
mov r1, ra5
sub.setf -, ra8, r1
brr.allz -, :store_results
nop
nop
nop

# Build tag(run_id, block_id, iter, logical_warp_id, lane).
mov r2, 0xb0000000
mov r3, ra1
shl r3, r3, 12
shl r3, r3, 12
or r2, r2, r3
mov r3, ra2
shl r3, r3, 12
shl r3, r3, 8
or r2, r2, r3
mov r3, ra8
shl r3, r3, 12
or r2, r2, r3
mov r3, ra3
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

# Write this logical warp tag to vpm_base_row + logical_warp_id.
mov r1, ra6
add r1, r1, ra3
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# Barrier barrier_after_writes: block 0 uses semaphores 0..3, block 1 uses 4..7.
sub.setf -, ra2, 0
brr.allz -, :barrier_after_writes_base0
nop
nop
nop
sub.setf -, ra2, 1
brr.allz -, :barrier_after_writes_base4
nop
nop
nop
brr -, :barrier_after_writes_base0
nop
nop
nop
:barrier_after_writes_base0
# Four-semaphore reusable barrier.  Leader is logical_warp_id 0.
sub.setf -, ra3, 0
brr.allz -, :barrier_after_writes_leader_b0
nop
nop
nop
:barrier_after_writes_nonleader_b0
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_after_writes_done
nop
nop
nop
:barrier_after_writes_leader_b0
:barrier_after_writes_arrive_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_arrive_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_release_b0_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_after_writes_arrive_b0_loop
nop
nop
nop
:barrier_after_writes_release_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_release_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_depart_b0_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_after_writes_release_b0_loop
nop
nop
nop
:barrier_after_writes_depart_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_depart_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_reset_b0_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_after_writes_depart_b0_loop
nop
nop
nop
:barrier_after_writes_reset_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_reset_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_after_writes_reset_b0_loop
nop
nop
nop
:barrier_after_writes_base4
# Four-semaphore reusable barrier.  Leader is logical_warp_id 0.
sub.setf -, ra3, 0
brr.allz -, :barrier_after_writes_leader_b4
nop
nop
nop
:barrier_after_writes_nonleader_b4
srel -, 4
sacq -, 5
srel -, 6
sacq -, 7
brr -, :barrier_after_writes_done
nop
nop
nop
:barrier_after_writes_leader_b4
:barrier_after_writes_arrive_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_arrive_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_release_b4_init
nop
nop
nop
sacq -, 4
sub r3, r3, 1
brr -, :barrier_after_writes_arrive_b4_loop
nop
nop
nop
:barrier_after_writes_release_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_release_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_depart_b4_init
nop
nop
nop
srel -, 5
sub r3, r3, 1
brr -, :barrier_after_writes_release_b4_loop
nop
nop
nop
:barrier_after_writes_depart_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_depart_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_reset_b4_init
nop
nop
nop
sacq -, 6
sub r3, r3, 1
brr -, :barrier_after_writes_depart_b4_loop
nop
nop
nop
:barrier_after_writes_reset_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_writes_reset_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_writes_done
nop
nop
nop
srel -, 7
sub r3, r3, 1
brr -, :barrier_after_writes_reset_b4_loop
nop
nop
nop
:barrier_after_writes_done

# Read every peer row in this block and accumulate a seen mask.  A peer
# is counted only if all 16 SIMD lanes match the lane-distinctive tag.
mov rb20, 0
mov rb21, 1
mov rb22, 0

:peer_loop
sub.setf -, rb20, ra4
brr.allz -, :peer_done
nop
nop
nop

# Read peer row vpm_base_row + peer_warp.
add r1, ra6, rb20
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, r1
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0

# Build expected peer tag into r1.
mov r1, 0xb0000000
mov r3, ra1
shl r3, r3, 12
shl r3, r3, 12
or r1, r1, r3
mov r3, ra2
shl r3, r3, 12
shl r3, r3, 8
or r1, r1, r3
mov r3, ra8
shl r3, r3, 12
or r1, r1, r3
mov r3, rb20
shl r3, r3, 8
or r1, r1, r3
or r1, r1, elem_num

# If every lane matches, include this peer bit; otherwise count one
# vector mismatch for this reader/peer/iteration.
sub.setf -, r2, r1
brr.allz -, :peer_match
nop
nop
nop

add ra11, ra11, 1
brr -, :peer_next
nop
nop
nop

:peer_match
mov r1, rb22
or r1, r1, rb21
mov rb22, r1

:peer_next
# VC4 small-immediate ALU instructions encode the immediate in the B-read
# field, so they cannot also read a source from regfile B.  Move the B-reg
# loop state through an accumulator before using the immediate 1.
mov r3, rb21
shl r3, r3, 1
mov rb21, r3
mov r3, rb20
add r3, r3, 1
mov rb20, r3
brr -, :peer_loop
nop
nop
nop

:peer_done
mov r1, rb22
or ra9, ra9, r1
and ra10, ra10, r1

# Barrier barrier_after_reads: block 0 uses semaphores 0..3, block 1 uses 4..7.
sub.setf -, ra2, 0
brr.allz -, :barrier_after_reads_base0
nop
nop
nop
sub.setf -, ra2, 1
brr.allz -, :barrier_after_reads_base4
nop
nop
nop
brr -, :barrier_after_reads_base0
nop
nop
nop
:barrier_after_reads_base0
# Four-semaphore reusable barrier.  Leader is logical_warp_id 0.
sub.setf -, ra3, 0
brr.allz -, :barrier_after_reads_leader_b0
nop
nop
nop
:barrier_after_reads_nonleader_b0
srel -, 0
sacq -, 1
srel -, 2
sacq -, 3
brr -, :barrier_after_reads_done
nop
nop
nop
:barrier_after_reads_leader_b0
:barrier_after_reads_arrive_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_arrive_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_release_b0_init
nop
nop
nop
sacq -, 0
sub r3, r3, 1
brr -, :barrier_after_reads_arrive_b0_loop
nop
nop
nop
:barrier_after_reads_release_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_release_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_depart_b0_init
nop
nop
nop
srel -, 1
sub r3, r3, 1
brr -, :barrier_after_reads_release_b0_loop
nop
nop
nop
:barrier_after_reads_depart_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_depart_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_reset_b0_init
nop
nop
nop
sacq -, 2
sub r3, r3, 1
brr -, :barrier_after_reads_depart_b0_loop
nop
nop
nop
:barrier_after_reads_reset_b0_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_reset_b0_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_done
nop
nop
nop
srel -, 3
sub r3, r3, 1
brr -, :barrier_after_reads_reset_b0_loop
nop
nop
nop
:barrier_after_reads_base4
# Four-semaphore reusable barrier.  Leader is logical_warp_id 0.
sub.setf -, ra3, 0
brr.allz -, :barrier_after_reads_leader_b4
nop
nop
nop
:barrier_after_reads_nonleader_b4
srel -, 4
sacq -, 5
srel -, 6
sacq -, 7
brr -, :barrier_after_reads_done
nop
nop
nop
:barrier_after_reads_leader_b4
:barrier_after_reads_arrive_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_arrive_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_release_b4_init
nop
nop
nop
sacq -, 4
sub r3, r3, 1
brr -, :barrier_after_reads_arrive_b4_loop
nop
nop
nop
:barrier_after_reads_release_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_release_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_depart_b4_init
nop
nop
nop
srel -, 5
sub r3, r3, 1
brr -, :barrier_after_reads_release_b4_loop
nop
nop
nop
:barrier_after_reads_depart_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_depart_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_reset_b4_init
nop
nop
nop
sacq -, 6
sub r3, r3, 1
brr -, :barrier_after_reads_depart_b4_loop
nop
nop
nop
:barrier_after_reads_reset_b4_init
mov r3, ra4
sub r3, r3, 1
:barrier_after_reads_reset_b4_loop
sub.setf -, r3, 0
brr.allz -, :barrier_after_reads_done
nop
nop
nop
srel -, 7
sub r3, r3, 1
brr -, :barrier_after_reads_reset_b4_loop
nop
nop
nop
:barrier_after_reads_done

add ra8, ra8, 1
brr -, :iter_loop
nop
nop
nop

:store_results
# Store compact per-warp vectors. Offsets match qpu_barrier_warp_result.
# offset 16: physical_qpu_report[16]
# offset 80: all_seen_mask_by_lane[16]
# offset 144: always_seen_mask_by_lane[16]
# offset 208: mismatch_count_by_lane[16]

# Store physical_qpu_report vector to result_warp_ptr + 16.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 63
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 63
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov r1, 16
add r1, ra12, r1
mov vw_addr, r1
read vw_wait
mov mutex_rel, 0

# Store all_seen_mask_by_lane vector to result_warp_ptr + 80.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 63
add vw_setup, r3, r1
mov vpm, ra9
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 63
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov r1, 80
add r1, ra12, r1
mov vw_addr, r1
read vw_wait
mov mutex_rel, 0

# Store always_seen_mask_by_lane vector to result_warp_ptr + 144.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 63
add vw_setup, r3, r1
mov vpm, ra10
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 63
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov r1, 144
add r1, ra12, r1
mov vw_addr, r1
read vw_wait
mov mutex_rel, 0

# Store mismatch_count_by_lane vector to result_warp_ptr + 208.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 63
add vw_setup, r3, r1
mov vpm, ra11
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 63
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov r1, 208
add r1, ra12, r1
mov vw_addr, r1
read vw_wait
mov mutex_rel, 0

:end
thrend
nop
nop
