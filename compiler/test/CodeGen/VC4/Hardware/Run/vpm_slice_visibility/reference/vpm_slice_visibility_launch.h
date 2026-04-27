#ifndef VPM_SLICE_VISIBILITY_LAUNCH_H
#define VPM_SLICE_VISIBILITY_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define VPM_SLICE_VISIBILITY_LANES 16u
#define VPM_SLICE_VISIBILITY_MAX_QPUS 16u
#define VPM_SLICE_VISIBILITY_MAX_VISIBILITY_PAIRS 5u
#define VPM_SLICE_VISIBILITY_MAX_COLLISION_RUNS 6u
#define VPM_SLICE_VISIBILITY_TEST_ROW 17u
#define VPM_SLICE_VISIBILITY_VPM_URSV_4K 16u
#define VPM_SLICE_VISIBILITY_SENTINEL 0xdeadbeefu

struct vpm_slice_visibility_sanity_result
{
    uint32_t qpu_report[VPM_SLICE_VISIBILITY_LANES];
    uint32_t observed[VPM_SLICE_VISIBILITY_LANES];
    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t timeout;
};

struct vpm_slice_visibility_pair_result
{
    uint32_t writer_qpu;
    uint32_t reader_qpu;
    uint32_t writer_slice;
    uint32_t reader_slice;
    uint32_t trial;
    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t timeout;
    uint32_t observed[VPM_SLICE_VISIBILITY_LANES];
    uint32_t reader_report[VPM_SLICE_VISIBILITY_LANES];
    uint32_t writer_report[VPM_SLICE_VISIBILITY_LANES];
};

struct vpm_slice_visibility_collision_result
{
    uint32_t qpu_a;
    uint32_t qpu_b;
    uint32_t slice_a;
    uint32_t slice_b;
    uint32_t order;
    uint32_t trial;
    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t timeout;
    uint32_t observed_by_a[VPM_SLICE_VISIBILITY_LANES];
    uint32_t observed_by_b[VPM_SLICE_VISIBILITY_LANES];
    uint32_t qpu_a_report[VPM_SLICE_VISIBILITY_LANES];
    uint32_t qpu_b_report[VPM_SLICE_VISIBILITY_LANES];
};

struct vpm_slice_visibility_results
{
    uint32_t ident1;
    uint32_t vpmsz_field;
    uint32_t vpm_kib;
    uint32_t tmus_per_slice;
    uint32_t qpus_per_slice;
    uint32_t num_slices;
    uint32_t revision;
    uint32_t num_qpus;
    uint32_t vpmbase_written;
    uint32_t vpmbase_readback;
    uint32_t srqcs_after_last_run;
    uint32_t timeouts;

    struct vpm_slice_visibility_sanity_result sanity;

    uint32_t visibility_pair_count;
    struct vpm_slice_visibility_pair_result
        visibility[VPM_SLICE_VISIBILITY_MAX_VISIBILITY_PAIRS];

    uint32_t collision_run_count;
    struct vpm_slice_visibility_collision_result
        collision[VPM_SLICE_VISIBILITY_MAX_COLLISION_RUNS];
};

/*
 * Public semantic launcher API for vpm_slice_visibility.
 *
 * Runs the exploratory VPM sanity, representative visibility-pair, and
 * representative collision-pair probes described in the handwritten harness.
 * The public API exposes only the runtime handle and an output result object.
 * It does not expose raw uniform streams or per-call scheduler internals.
 */
int vpm_slice_visibility_launch(
    struct vc4_runtime *rt,
    struct vpm_slice_visibility_results *out);

#endif
