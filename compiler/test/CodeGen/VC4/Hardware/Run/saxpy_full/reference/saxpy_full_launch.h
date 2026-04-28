#ifndef SAXPY_FULL_LAUNCH_H
#define SAXPY_FULL_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Public semantic launcher API for saxpy_full.
 *
 * Computes y[i] = alpha * x[i] + y[i] for every i in [0, n).
 *
 * Unlike saxpy_16, this reference has tail_safe semantics: n may be zero,
 * less than one vector, non-multiple-of-16, non-multiple-of-active-QPUs, or
 * larger than one full active-QPU round.
 *
 * Runtime/setup discipline for this test:
 *   - saxpy_full_prepare() performs the single GPU allocation for this kernel
 *     and copies the assembled qasm code to that allocation exactly once.
 *   - saxpy_full_launch() only updates the already-allocated payload/uniforms
 *     and queues the already-resident shader code.
 *   - The bare-metal hardware test intentionally keeps this allocation live
 *     until the Pi is power-cycled; this avoids repeated alloc/lock/unlock/free
 *     cycles while exercising many different n values in one boot.
 *
 * The public launch API exposes only semantic arguments plus the runtime
 * handle. It does not expose qpu_id, num_qpus, raw uniform streams, VPM rows,
 * TMU details, or scheduler internals.
 */
int saxpy_full_prepare(struct vc4_runtime *rt, uint32_t max_n);

int saxpy_full_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *y,
    float alpha,
    uint32_t n);

/* Optional cleanup for callers that want an explicit one-time teardown. */
void saxpy_full_release(struct vc4_runtime *rt);

/* Test diagnostics: these do not expose raw scheduler or uniform internals. */
uint32_t saxpy_full_runtime_allocations(void);
uint32_t saxpy_full_runtime_launches(void);
uint32_t saxpy_full_runtime_capacity(void);

#endif
