#ifndef SAXPY_BASIC_LAUNCH_H
#define SAXPY_BASIC_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Compiler-owned launcher interface for the saxpy_basic hardware
 * golden.
 *
 * Public contract:
 *   int saxpy_basic_launch(
 *       struct vc4_runtime *rt,
 *       const float *x,
 *       float *y,
 *       float alpha,
 *       uint32_t n);
 *
 * The public API exposes only semantic kernel arguments plus a runtime handle.
 * It does not expose qpu_id, num_qpus, uniform packing, bus addresses, VPM rows,
 * or scheduler internals.
 *
 * This small golden runs exactly one 16-lane vector per active QPU and therefore
 * requires n == active_qpus * lane_width. That is a local test policy, not a
 * global VC4 codegen rule.
 */
int saxpy_basic_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *y,
    float alpha,
    uint32_t n);

#endif
