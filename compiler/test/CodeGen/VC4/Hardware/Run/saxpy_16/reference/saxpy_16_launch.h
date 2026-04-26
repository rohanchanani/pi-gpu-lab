#ifndef SAXPY_16_LAUNCH_H
#define SAXPY_16_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Public semantic launcher API for saxpy_16.
 *
 * Computes y[i] = alpha * x[i] + y[i] for n elements.
 *
 * The public API exposes only semantic arguments plus the runtime handle.  It
 * does not expose qpu_id, num_qpus, raw uniform streams, VPM rows, TMU details,
 * or scheduler internals.
 *
 * The generated-style launcher does not validate semantic kernel argument
 * values such as n.  The checked hardware harness supplies an exact-multiple
 * n for this golden; tail behavior is a kernel/upstream contract, not a
 * launcher guard.
 */
int saxpy_16_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *y,
    float alpha,
    uint32_t n);

#endif
