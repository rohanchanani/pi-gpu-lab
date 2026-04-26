#include "rpi.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "tmu_read_nop_write_launch.h"
#include "tmu_read_nop_writeshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 5

struct tmu_read_nop_write_launch_state
{
    uint32_t code[sizeof(tmu_read_nop_writeshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    uint32_t gpu_input[TMU_READ_NOP_WRITE_MAX_WORDS];
    uint32_t gpu_result[TMU_READ_NOP_WRITE_MAX_WORDS];
};

int tmu_read_nop_write_launch(struct vc4_runtime *rt,
                              const uint32_t *input,
                              uint32_t *result,
                              uint32_t words)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!input || !result)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    uint32_t laneWidth = vc4_runtime_lane_width();
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;
    if (laneWidth != TMU_READ_NOP_WRITE_WORDS_PER_QPU)
        return -1;
    if (words != activeQpus * TMU_READ_NOP_WRITE_WORDS_PER_QPU)
        return -1;
    if (words > TMU_READ_NOP_WRITE_MAX_WORDS)
        return -1;

    uint32_t handle = mem_alloc(sizeof(struct tmu_read_nop_write_launch_state),
                                4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct tmu_read_nop_write_launch_state *state =
        (volatile struct tmu_read_nop_write_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, tmu_read_nop_writeshader, sizeof state->code);
    memcpy((void *)state->gpu_input, input, words * sizeof(uint32_t));
    memset((void *)state->gpu_result, 0, words * sizeof(uint32_t));

    uint32_t gpuInputAddr = GPU_BASE + (uint32_t)state->gpu_input;
    uint32_t gpuResultAddr = GPU_BASE + (uint32_t)state->gpu_result;

    for (uint32_t qpu = 0; qpu < activeQpus; ++qpu)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] input base address
         *   [1] result base address
         *   [2] word count
         *   [3] qpu_id
         *   [4] num_qpus
         */
        state->unif[qpu][0] = gpuInputAddr;
        state->unif[qpu][1] = gpuResultAddr;
        state->unif[qpu][2] = words;
        state->unif[qpu][3] = qpu;
        state->unif[qpu][4] = activeQpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             activeQpus);

    memcpy(result, (const void *)state->gpu_result,
           words * sizeof(uint32_t));

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
