#ifndef SAXPY_LAUNCH_H
#define SAXPY_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

/*
 * Compiler-owned launcher interface for the current narrow SAXPY slice.
 *
 * Public contract:
 *   int saxpy_launch(struct vc4_runtime *rt, float *x, float *y, float a, uint32_t n);
 *
 * This interface exposes only semantic kernel arguments plus a runtime handle.
 * It does not expose qpu_id, num_qpus, uniform packing, or other ABI internals.
 * Launch policy such as the active QPU count comes from the generic runtime.
 */
int saxpy_launch(struct vc4_runtime *rt, float *x, float *y, float a, uint32_t n);

#endif
