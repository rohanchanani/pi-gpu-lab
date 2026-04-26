#ifndef MEMORY_OUTPUT_LAUNCH_H
#define MEMORY_OUTPUT_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

enum {
    MEMORY_OUTPUT_WORDS_PER_QPU = 16,
    MEMORY_OUTPUT_MAX_WORDS =
        VC4_RUNTIME_MAX_QPUS * MEMORY_OUTPUT_WORDS_PER_QPU
};

/*
 * Public reference-launcher contract:
 *
 *   int memory_output_launch(struct vc4_runtime *rt,
 *                            uint32_t *out,
 *                            uint32_t out_words);
 *
 * The launcher writes active_qpus * 16 u32 words to `out`.
 * It exposes only the semantic output buffer, not qpu_id/num_qpus or
 * physical uniform packing.
 */
int memory_output_launch(struct vc4_runtime *rt,
                         uint32_t *out,
                         uint32_t out_words);

#endif
