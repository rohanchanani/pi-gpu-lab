#ifndef QPU_NUM_REGISTER_LAUNCH_H
#define QPU_NUM_REGISTER_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define QPU_NUM_REGISTER_LAUNCHES 16u
#define QPU_NUM_REGISTER_LANE_WIDTH 16u
#define QPU_NUM_REGISTER_SENTINEL 0xdeadbeefu

/*
 * Public semantic launcher API for qpu_num_register.
 *
 * Queues exactly 16 VC4 QPU user-program requests. Each request observes the
 * hardware qpu_num / QPU_NUMBER register and writes one semantic result to
 * out[launch].
 *
 * The public API exposes only the semantic output buffer plus the runtime
 * handle. It does not expose qpu_id, num_qpus, raw uniform streams, VPM rows,
 * or V3D scheduler registers.
 */
int qpu_num_register_launch(
    struct vc4_runtime *rt,
    uint32_t out[QPU_NUM_REGISTER_LAUNCHES]);

#endif
