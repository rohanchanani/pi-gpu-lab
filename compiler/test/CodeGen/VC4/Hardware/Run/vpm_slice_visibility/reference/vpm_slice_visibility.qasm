.include "../share/vc4inc/vc4.qinc"

# vpm_slice_visibility reference kernel.
#
# Exploratory VC4 QPU VPM storage-visibility smoke test.
#
# mode 0: single-QPU VPM sanity check
#   uniforms after mode:
#     [1] qpu_report_vector_bus
#     [2] observed_vector_bus
#     [3] test_row
#
# mode 1: ordered writer/reader visibility pair
#   uniforms after mode:
#     [1] writer_qpu
#     [2] reader_qpu
#     [3] trial
#     [4] test_row
#     [5] observed_vector_bus
#     [6] reader_report_vector_bus
#     [7] writer_report_vector_bus
#
# mode 2: same-address collision pair
#   uniforms after mode:
#     [1] qpu_a
#     [2] qpu_b
#     [3] order                # 0 = A then B, 1 = B then A
#     [4] test_row
#     [5] observed_by_a_bus
#     [6] observed_by_b_bus
#     [7] qpu_a_report_vector_bus
#     [8] qpu_b_report_vector_bus
#
# The storage tests hold the global QPU mutex around VPM setup/access and
# around VPM->VDW result stores.  Ordering between the two active QPUs uses
# QPU semaphore instructions only.  The final thread-end instruction and its
# two delay slots do not touch uniforms, VPM, VDR, VDW, or regfile address 14.

mov ra0, unif
nop

# Dispatch on mode.  The branches are deliberately separated by non-branch
# instructions and each branch has the required three delay-slot instructions.
sub.setf -, ra0, 0
brr.allz -, :sanity
nop
nop
nop

sub.setf -, ra0, 1
brr.allz -, :visibility
nop
nop
nop

sub.setf -, ra0, 2
brr.allz -, :collision
nop
nop
nop

brr -, :end
nop
nop
nop

:sanity
# Uniforms: ra1=qpu_report_bus, ra2=observed_bus, ra3=test_row.
mov ra1, unif
mov ra2, unif
mov ra3, unif

mov r0, qpu_num

# r2[lane] = 0xD5000000 | elem_num.
mov r2, 0xd5000000
or r2, r2, elem_num

# Write the sanity vector to TEST_ROW, then read the same row back.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra3
mov vpm, r2
read vw_wait
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra3
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0

# Store qpu_num vector to host at ra1 using RESULT_ROW_A = 60.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 60
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 60
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra1
read vw_wait
mov mutex_rel, 0

# Store observed sanity vector to host at ra2 using RESULT_ROW_ONE = 62.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 62
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 62
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra2
read vw_wait
mov mutex_rel, 0

brr -, :end
nop
nop
nop

:visibility
# Uniforms:
#   ra1=writer_qpu, ra2=reader_qpu, ra3=trial, ra4=test_row,
#   ra5=observed_bus, ra6=reader_report_bus, ra7=writer_report_bus.
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif

mov r0, qpu_num

sub.setf -, r0, ra1
brr.allz -, :vis_writer
nop
nop
nop

sub.setf -, r0, ra2
brr.allz -, :vis_reader
nop
nop
nop

# An unexpected QPU should not happen if SQRSV0/1 reservations worked.  Exit
# rather than participating in the semaphore protocol and corrupting storage.
brr -, :end
nop
nop
nop

:vis_writer
# Start barrier: writer and reader both report ready, then consume the other
# side's ready token.  S_WRITER_READY=0, S_READER_READY=1.
srel -, 0
sacq -, 1

# Wait until the reader has cleared TEST_ROW.  S_CLEARED=2.
sacq -, 2

# r2[lane] = visibility tag for writer_qpu/trial/lane.
mov r2, 0xa5000000
and r3, ra3, 15
shl r3, r3, 8
shl r3, r3, 12
or r2, r2, r3
and r3, ra1, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

# Write writer tag to TEST_ROW under the global mutex.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# Store writer's reported qpu_num vector to host at ra7.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 60
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 60
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra7
read vw_wait
mov mutex_rel, 0

# Release reader to observe the row, then wait for reader completion.
# S_WRITTEN=3, S_READER_DONE=4.
srel -, 3
sacq -, 4

brr -, :end
nop
nop
nop

:vis_reader
# Start barrier.  S_READER_READY=1, S_WRITER_READY=0.
srel -, 1
sacq -, 0

# Clear TEST_ROW under the global mutex before allowing the writer to write.
mov r2, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# Tell the writer the row is cleared, then wait for the writer's tag.
srel -, 2
sacq -, 3

# Read TEST_ROW under the global mutex.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra4
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0

# Store observed vector to host at ra5 using RESULT_ROW_ONE = 62.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 62
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 62
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra5
read vw_wait
mov mutex_rel, 0

# Store reader's reported qpu_num vector to host at ra6.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 61
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 61
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra6
read vw_wait
mov mutex_rel, 0

srel -, 4

brr -, :end
nop
nop
nop

:collision
# Uniforms:
#   ra1=qpu_a, ra2=qpu_b, ra3=order, ra4=test_row,
#   ra5=observed_by_a_bus, ra6=observed_by_b_bus,
#   ra7=qpu_a_report_bus, ra8=qpu_b_report_bus.
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif
mov ra5, unif
mov ra6, unif
mov ra7, unif
mov ra8, unif

mov r0, qpu_num

sub.setf -, r0, ra1
brr.allz -, :col_a
nop
nop
nop

sub.setf -, r0, ra2
brr.allz -, :col_b
nop
nop
nop

brr -, :end
nop
nop
nop

:col_a
# Start barrier: S_A_READY=0, S_B_READY=1.
srel -, 0
sacq -, 1

# Clear this QPU's view of TEST_ROW.
mov r2, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# Clear barrier: S_A_CLEAR=2, S_B_CLEAR=3.
srel -, 2
sacq -, 3

sub.setf -, ra3, 0
brr.allz -, :col_a_order0
nop
nop
nop
brr -, :col_a_order1
nop
nop
nop

:col_b
# Start barrier: S_B_READY=1, S_A_READY=0.
srel -, 1
sacq -, 0

# Clear this QPU's view of TEST_ROW.
mov r2, 0
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# Clear barrier: S_B_CLEAR=3, S_A_CLEAR=2.
srel -, 3
sacq -, 2

sub.setf -, ra3, 0
brr.allz -, :col_b_order0
nop
nop
nop
brr -, :col_b_order1
nop
nop
nop

:col_a_order0
# Order 0: A writes first, B writes second.
# Build tag_a = 0xC5000000 | ((order&15)<<20) | ((qpu_a&15)<<8) | lane.
mov r2, 0xc5000000
and r3, ra3, 15
shl r3, r3, 8
shl r3, r3, 12
or r2, r2, r3
and r3, ra1, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

# S_A_WROTE=4, S_B_WROTE=5.
srel -, 4
sacq -, 5

brr -, :col_a_read_store
nop
nop
nop

:col_b_order0
# Order 0: wait for A, then B writes second.
sacq -, 4

mov r2, 0xc5000000
and r3, ra3, 15
shl r3, r3, 8
shl r3, r3, 12
or r2, r2, r3
and r3, ra2, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

srel -, 5

brr -, :col_b_read_store
nop
nop
nop

:col_a_order1
# Order 1: B writes first, A writes second.
sacq -, 5

mov r2, 0xc5000000
and r3, ra3, 15
shl r3, r3, 8
shl r3, r3, 12
or r2, r2, r3
and r3, ra1, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

srel -, 4

brr -, :col_a_read_store
nop
nop
nop

:col_b_order1
# Order 1: B writes first, then waits for A to write second.
mov r2, 0xc5000000
and r3, ra3, 15
shl r3, r3, 8
shl r3, r3, 12
or r2, r2, r3
and r3, ra2, 15
shl r3, r3, 8
or r2, r2, r3
or r2, r2, elem_num

read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra4
mov vpm, r2
read vw_wait
mov mutex_rel, 0

srel -, 5
sacq -, 4

brr -, :col_b_read_store
nop
nop
nop

:col_a_read_store
# Read TEST_ROW into r2.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra4
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0

# Store observed_by_a at ra5 using RESULT_ROW_A = 60.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 60
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 60
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra5
read vw_wait
mov mutex_rel, 0

# Store qpu_a report vector at ra7.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 61
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 61
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra7
read vw_wait
mov mutex_rel, 0

# Store-completion barrier: S_A_STORED=6, S_B_STORED=7.
srel -, 6
sacq -, 7

brr -, :end
nop
nop
nop

:col_b_read_store
# Read TEST_ROW into r2.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
add vr_setup, r3, ra4
nop
nop
nop
mov r2, vpm
mov mutex_rel, 0

# Store observed_by_b at ra6 using RESULT_ROW_B = 61.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 61
add vw_setup, r3, r1
mov vpm, r2
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 61
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra6
read vw_wait
mov mutex_rel, 0

# Store qpu_b report vector at ra8.
read mutex_acq
mov r3, vpm_setup(1, 1, h32(0))
mov r1, 60
add vw_setup, r3, r1
mov vpm, r0
read vw_wait
mov r3, vdw_setup_1(0)
mov vw_setup, r3
mov r1, 60
shl r1, r1, 7
mov r3, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r3, r1
mov vw_addr, ra8
read vw_wait
mov mutex_rel, 0

# Store-completion barrier: S_B_STORED=7, S_A_STORED=6.
srel -, 7
sacq -, 6

brr -, :end
nop
nop
nop

:end
thrend
nop
nop
