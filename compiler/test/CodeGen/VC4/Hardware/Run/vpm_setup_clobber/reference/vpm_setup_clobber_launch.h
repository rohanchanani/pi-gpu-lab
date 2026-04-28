#ifndef VPM_SETUP_CLOBBER_LAUNCH_H
#define VPM_SETUP_CLOBBER_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define VPM_SETUP_CLOBBER_LANES 16u
#define VPM_SETUP_CLOBBER_MAX_QPUS 16u
#define VPM_SETUP_CLOBBER_VPM_URSV_4K 16u
#define VPM_SETUP_CLOBBER_CLOBBER_ROWA 17u
#define VPM_SETUP_CLOBBER_CLOBBER_ROWB 18u
#define VPM_SETUP_CLOBBER_SENTINEL 0xdeadbeefu
#define VPM_SETUP_CLOBBER_TIMEOUT_USEC 2000000u

#define VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER 0u
#define VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER 1u
#define VPM_SETUP_CLOBBER_CLASS_BOTH 2u
#define VPM_SETUP_CLOBBER_CLASS_OTHER 3u

struct vpm_setup_clobber_topology {
uint32_t ident1;
uint32_t vpmsz_field;
uint32_t vpm_kib;
uint32_t qpus_per_slice;
uint32_t num_slices;
uint32_t num_qpus;
uint32_t num_semaphores;
uint32_t vpmbase_written;
uint32_t vpmbase_readback;
};

struct vpm_setup_clobber_record {
uint32_t trial;
uint32_t qpu_a;
uint32_t qpu_b;
uint32_t slice_a;
uint32_t slice_b;

uint32_t row_a[VPM_SETUP_CLOBBER_LANES];
uint32_t row_b[VPM_SETUP_CLOBBER_LANES];
uint32_t qpu_a_report[VPM_SETUP_CLOBBER_LANES];
uint32_t qpu_b_report[VPM_SETUP_CLOBBER_LANES];

uint32_t row_a_match;
uint32_t row_b_match;
uint32_t classification;
uint32_t qpu_mismatches;

uint32_t errstat_before;
uint32_t errstat_after;
uint32_t timeout;

};

struct vpm_setup_clobber_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;

struct vpm_setup_clobber_topology topology;

uint32_t runtime_allocations;
uint32_t runtime_launches;
uint32_t srqcs_after_last_run;

};

/*

Diagnostic low-level public API for vpm_setup_clobber.

This test is explicitly a topology/hardware-state litmus, so the API exposes

intended physical QPU pair values. It still keeps uniform packing, QPU

reservation programming, GPU code address, scheduler registers, and result

scratch placement private inside the launcher implementation.
*/
int vpm_setup_clobber_prepare(
struct vc4_runtime *rt,
struct vpm_setup_clobber_state *state);

int vpm_setup_clobber_run_pair(
struct vpm_setup_clobber_state *state,
uint32_t qpu_a,
uint32_t qpu_b,
uint32_t trial,
struct vpm_setup_clobber_record *out);

void vpm_setup_clobber_shutdown(struct vpm_setup_clobber_state *state);

const char *vpm_setup_clobber_classification_name(uint32_t classification);

#endif

