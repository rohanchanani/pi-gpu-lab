#ifndef READ_NOP_WRITE_LAUNCH_H
#define READ_NOP_WRITE_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Reference launcher API for the read_nop_write hardware contract test.
 *
 * Public contract:
 *   int read_nop_write_launch(
 *       struct vc4_runtime *rt,
 *       const uint32_t *input,
 *       uint32_t *result,
 *       uint32_t n);
 *
 * This API exposes semantic arguments plus a runtime handle only. It does not
 * expose qpu_id, num_qpus, uniform packing, VPM rows, V3D scheduler registers,
 * or GPU bus addresses.
 *
 * Test-specific policy:
 *   n must equal active_qpus * vc4_runtime_lane_width().
 */
int read_nop_write_launch(
    struct vc4_runtime *rt,
    const uint32_t *input,
    uint32_t *result,
    uint32_t n);

#endif
