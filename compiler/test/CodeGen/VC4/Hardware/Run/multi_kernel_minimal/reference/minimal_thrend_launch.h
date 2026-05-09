#ifndef MINIMAL_THREND_LAUNCH_H
#define MINIMAL_THREND_LAUNCH_H

#include "mailbox.h"

/*
 * Public launcher for the minimal thread-end hardware contract test.
 *
 * Public contract:
 *   int minimal_thrend_launch(struct vc4_runtime *rt);
 *
 * The launcher exposes no kernel semantic arguments.  Internally, it still
 * packs one per-QPU physical uniform stream containing qpu_id and num_qpus, so
 * the test exercises the same public/physical ABI split as richer kernels.
 */
int minimal_thrend_launch(struct vc4_runtime *rt);

#endif /* MINIMAL_THREND_LAUNCH_H */
