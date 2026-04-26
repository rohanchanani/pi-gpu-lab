#ifndef SFU_RECIP_LAUNCH_H
#define SFU_RECIP_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Public semantic launcher API for sfu_recip.
 *
 * Computes out[i] = recip(x[i]) for this test's one-vector-per-active-QPU
 * policy. The handwritten harness supplies:
 *
 *   n == vc4_runtime_active_qpus(rt) * vc4_runtime_lane_width()
 *
 * The public API exposes only semantic arguments plus the runtime handle. It
 * does not expose qpu_id, num_qpus, raw uniform streams, GPU bus addresses,
 * VPM rows, SFU register details, or scheduler internals.
 */
int sfu_recip_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *out,
    uint32_t n);

#endif
