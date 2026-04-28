#ifndef MATMUL_BLOCKED_LAUNCH_H
#define MATMUL_BLOCKED_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Public semantic launcher API for matmul_blocked.
 *
 * Computes row-major C[M,N] = A[M,K] * B[K,N] for f32 matrices.
 *
 * CUDA-like cooperative-block shape used by this reference:
 *   - one resident block computes a 12-row x 16-column C tile;
 *   - one QPU request is one logical 16-lane warp;
 *   - logical_warp_id is supplied by the runtime as a uniform;
 *   - ELEMENT_NUMBER is the lane / column within the 16-column tile;
 *   - VPM rows are used as CUDA-like shared memory for a B K-tile;
 *   - a four-semaphore reusable barrier synchronizes the 12 resident warps.
 *
 * Runtime/setup discipline for this test:
 *   - matmul_blocked_prepare() performs the single GPU allocation for this
 *     kernel and copies the assembled qasm code to that allocation exactly once.
 *   - matmul_blocked_launch() only refreshes the already-allocated payload and
 *     uniform streams, then queues the already-resident shader code one tile
 *     wave at a time.
 *   - The bare-metal hardware test keeps the allocation live until the runner
 *     power-cycles the Pi.  This avoids repeated alloc/lock/unlock/free cycles.
 *
 * The public launch API exposes only semantic arguments plus the runtime
 * handle. It does not expose physical QPU IDs, raw uniform arrays, VPM rows,
 * semaphore IDs, or scheduler registers.
 */
int matmul_blocked_prepare(
    struct vc4_runtime *rt,
    uint32_t max_m,
    uint32_t max_n,
    uint32_t max_k);

int matmul_blocked_launch(
    struct vc4_runtime *rt,
    const float *a,
    const float *b,
    float *c,
    uint32_t m,
    uint32_t n,
    uint32_t k);

/* Optional one-time teardown. The hardware test intentionally does not call it
 * inside the multi-case loop. */
void matmul_blocked_release(struct vc4_runtime *rt);

/* Test diagnostics: these do not expose raw uniform streams. */
uint32_t matmul_blocked_runtime_allocations(void);
uint32_t matmul_blocked_runtime_launches(void);
uint32_t matmul_blocked_runtime_tile_waves(void);
uint32_t matmul_blocked_runtime_timeouts(void);
uint32_t matmul_blocked_runtime_errstat_relevant_changed(void);
uint32_t matmul_blocked_runtime_vpmbase_readback(void);
uint32_t matmul_blocked_runtime_ident1(void);
uint32_t matmul_blocked_runtime_srqcs_after_last_wave(void);
uint32_t matmul_blocked_runtime_capacity_m(void);
uint32_t matmul_blocked_runtime_capacity_n(void);
uint32_t matmul_blocked_runtime_capacity_k(void);

#endif
