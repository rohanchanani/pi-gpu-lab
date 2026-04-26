#ifndef TMU_READ_NOP_WRITE_LAUNCH_H
#define TMU_READ_NOP_WRITE_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

enum {
    TMU_READ_NOP_WRITE_WORDS_PER_QPU = 16,
    TMU_READ_NOP_WRITE_MAX_WORDS =
        VC4_RUNTIME_MAX_QPUS * TMU_READ_NOP_WRITE_WORDS_PER_QPU,
};

/*
 * Public semantic launcher contract.
 *
 * Copies `words` u32 values from `input` to `result` using one 16-lane vector
 * per active QPU. The reference kernel requires:
 *
 *   words == vc4_runtime_active_qpus(rt) * 16
 *
 * The launcher hides physical uniforms, qpu_id, num_qpus, GPU addresses, and
 * scheduler details.
 */
int tmu_read_nop_write_launch(struct vc4_runtime *rt,
                              const uint32_t *input,
                              uint32_t *result,
                              uint32_t words);

#endif
