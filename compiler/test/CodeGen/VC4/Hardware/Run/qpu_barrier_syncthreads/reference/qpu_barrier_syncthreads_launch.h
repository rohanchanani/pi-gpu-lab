#ifndef QPU_BARRIER_SYNCTHREADS_LAUNCH_H
#define QPU_BARRIER_SYNCTHREADS_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define QPU_BARRIER_LANES              16u
#define QPU_BARRIER_MAX_QPUS           16u
#define QPU_BARRIER_EXPECTED_QPUS      12u
#define QPU_BARRIER_VPM_URSV_4K        16u
#define QPU_BARRIER_VPM_ROWS           64u
#define QPU_BARRIER_ROW_WORDS          16u
#define QPU_BARRIER_SENTINEL           0xdeadbeefu
#define QPU_BARRIER_MAX_ITERS          64u
#define QPU_BARRIER_TIMEOUT_USEC       2000000u
#define QPU_BARRIER_MAX_RUNS           8u
#define QPU_BARRIER_MAX_WARPS          12u

#define QPU_BARRIER_MODE_SAME_SLICE_SMOKE       0u
#define QPU_BARRIER_MODE_CROSS_SLICE_0_1        1u
#define QPU_BARRIER_MODE_CROSS_SLICE_0_2        2u
#define QPU_BARRIER_MODE_FULL_BLOCK_STRESS      3u
#define QPU_BARRIER_MODE_TWO_BLOCK_PARTITION    4u

struct qpu_barrier_warp_result {
    uint32_t block_id;
    uint32_t logical_warp_id;
    uint32_t reserved0;
    uint32_t reserved1;

    uint32_t physical_qpu_report[QPU_BARRIER_LANES];
    uint32_t all_seen_mask_by_lane[QPU_BARRIER_LANES];
    uint32_t always_seen_mask_by_lane[QPU_BARRIER_LANES];
    uint32_t mismatch_count_by_lane[QPU_BARRIER_LANES];
};

struct qpu_barrier_run_result {
    uint32_t run_id;
    uint32_t mode;
    uint32_t block_count;
    uint32_t warps_per_block;
    uint32_t total_requests;
    uint32_t iterations;

    uint32_t sem_base;
    uint32_t vpm_base_row[4];
    uint32_t vpm_rows_per_block;

    uint32_t qpu_set_mask;
    uint32_t expected_qpu_mask;
    uint32_t observed_qpu_mask;

    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t timeout;

    uint32_t mismatch_count;
    uint32_t first_bad_iter;
    uint32_t first_bad_block;
    uint32_t first_bad_reader_warp;
    uint32_t first_bad_peer_warp;
    uint32_t first_bad_lane;
    uint32_t first_bad_observed;
    uint32_t first_bad_expected;

    struct qpu_barrier_warp_result warp[QPU_BARRIER_MAX_WARPS];
};

struct qpu_barrier_results {
    uint32_t ident1;
    uint32_t vpmsz_field;
    uint32_t vpm_kib;
    uint32_t qpus_per_slice;
    uint32_t num_slices;
    uint32_t num_qpus;
    uint32_t num_semaphores;
    uint32_t vpmbase_written;
    uint32_t vpmbase_readback;
    uint32_t srqcs_after_last_run;
    uint32_t timeouts;
    uint32_t errstat_changed_count;

    uint32_t run_count;
    struct qpu_barrier_run_result runs[QPU_BARRIER_MAX_RUNS];
};

/*
 * Public semantic launcher API for qpu_barrier_syncthreads.
 *
 * Runs the fixed hardware-grounded barrier validation modes and writes the
 * decoded topology and per-run diagnostics into out_results.  The public API
 * exposes only the runtime handle and semantic output buffer.  It does not
 * expose raw uniform streams, QPU reservation registers, semaphore IDs, VPM
 * rows, V3D scheduler internals, or physical QPU assignment controls.
 */
int qpu_barrier_syncthreads_launch(
    struct vc4_runtime *rt,
    struct qpu_barrier_results *out_results);

#endif
