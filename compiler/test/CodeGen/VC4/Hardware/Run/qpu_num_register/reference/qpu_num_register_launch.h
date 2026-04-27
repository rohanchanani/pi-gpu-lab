#ifndef QPU_NUM_REGISTER_LAUNCH_H
#define QPU_NUM_REGISTER_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * The documented QPU_NUMBER register is a 4-bit physical QPU id.  Keep a
 * 16-entry public result array, but in this replacement each slot is indexed by
 * target physical QPU, not by enqueue order.
 */
#define QPU_NUM_REGISTER_LAUNCHES 16u
#define QPU_NUM_REGISTER_LANE_WIDTH 16u
#define QPU_NUM_REGISTER_SENTINEL 0xdeadbeefu
#define QPU_NUM_REGISTER_ABSENT   0xffffffffu
#define QPU_NUM_REGISTER_MAX_QPUS 16u

struct qpu_num_register_debug
{
    uint32_t ident0;
    uint32_t ident1;
    uint32_t ident2;

    uint32_t qpus_per_slice;
    uint32_t num_slices;
    uint32_t hw_qpus;
    uint32_t active_qpus;
    uint32_t expected_mask;

    uint32_t seen_mask;
    uint32_t irq_mask;

    uint32_t sqrsv0_before;
    uint32_t sqrsv1_before;
    uint32_t sqrsv0_after;
    uint32_t sqrsv1_after;

    uint32_t vpm_base_before;
    uint32_t vpm_base_after;
    uint32_t dbqite_before;
    uint32_t dbqite_after;
    uint32_t srqcs_after;
    uint32_t errstat_before;
    uint32_t errstat_after;
};

/*
 * Targeted QPU_NUMBER test.
 *
 * This deliberately does NOT queue 16 tiny requests at once.  Instead it uses
 * V3D_SQRSV0/1 to reserve every physical QPU except the target, queues exactly
 * one user program, and records what QPU_NUMBER that target reports.
 *
 * Expected result on a 12-QPU VC4:
 *   out[0]  == 0
 *   ...
 *   out[11] == 11
 *   out[12]..out[15] == QPU_NUM_REGISTER_ABSENT
 */
int qpu_num_register_launch(
    struct vc4_runtime *rt,
    uint32_t out[QPU_NUM_REGISTER_LAUNCHES]);

const struct qpu_num_register_debug *qpu_num_register_last_debug(void);

#endif
