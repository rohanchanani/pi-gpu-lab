#ifndef MATMUL_NAIVE_LAUNCH_H
#define MATMUL_NAIVE_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Public semantic launcher API for matmul_naive.
 *
 * Computes row-major C[M,N] = A[M,K] * B[K,N] for f32 matrices.
 *
 * Runtime/setup discipline for this test:
 *   - matmul_naive_prepare() performs the single GPU allocation for this
 *     kernel and copies the assembled qasm code to that allocation exactly
 *     once.
 *   - matmul_naive_launch() only updates the already-allocated payload and
 *     uniform streams for the current M/N/K, then queues the already-resident
 *     shader code.
 *   - The bare-metal hardware test intentionally keeps this allocation live
 *     until the Pi is power-cycled; this avoids repeated alloc/lock/unlock/free
 *     cycles while exercising many matrix shapes in one boot.
 *
 * The public launch API exposes only semantic arguments plus the runtime
 * handle. It does not expose qpu_id, num_qpus, raw uniform streams, private
 * B guard padding, VPM rows, TMU details, or scheduler internals.
 */
int matmul_naive_prepare(
    struct vc4_runtime *rt,
    uint32_t max_m,
    uint32_t max_n,
    uint32_t max_k);

int matmul_naive_launch(
    struct vc4_runtime *rt,
    const float *a,
    const float *b,
    float *c,
    uint32_t m,
    uint32_t n,
    uint32_t k);

/* Optional cleanup for callers that want an explicit one-time teardown. */
void matmul_naive_release(struct vc4_runtime *rt);

/* Test diagnostics: these do not expose raw scheduler or uniform internals. */
uint32_t matmul_naive_runtime_allocations(void);
uint32_t matmul_naive_runtime_launches(void);
uint32_t matmul_naive_runtime_capacity_m(void);
uint32_t matmul_naive_runtime_capacity_n(void);
uint32_t matmul_naive_runtime_capacity_k(void);

#endif
