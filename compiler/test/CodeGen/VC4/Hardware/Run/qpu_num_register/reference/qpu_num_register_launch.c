#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "qpu_num_register_launch.h"
#include "qpu_num_registershader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 1

struct qpu_num_register_launch_state
{
    uint32_t code[sizeof(qpu_num_registershader) / sizeof(uint32_t)];
    uint32_t unif[QPU_NUM_REGISTER_LAUNCHES][NUM_UNIFS];
    uint32_t unif_ptr[QPU_NUM_REGISTER_LAUNCHES];
    uint32_t handle;
    uint32_t payload[];
};

static size_t qpu_num_register_state_size(void)
{
    return offsetof(struct qpu_num_register_launch_state, payload) +
           (size_t)QPU_NUM_REGISTER_LAUNCHES *
               QPU_NUM_REGISTER_LANE_WIDTH * sizeof(uint32_t);
}

static uint32_t *qpu_num_register_out_ptr(
    volatile struct qpu_num_register_launch_state *state)
{
    return (uint32_t *)state->payload;
}

int qpu_num_register_launch(
    struct vc4_runtime *rt,
    uint32_t out[QPU_NUM_REGISTER_LAUNCHES])
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!out)
        return -1;

    size_t allocSize = qpu_num_register_state_size();
    uint32_t handle = mem_alloc(allocSize, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct qpu_num_register_launch_state *state =
        (volatile struct qpu_num_register_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, qpu_num_registershader, sizeof state->code);

    uint32_t *gpuOut = qpu_num_register_out_ptr(state);
    for (uint32_t i = 0;
         i < QPU_NUM_REGISTER_LAUNCHES * QPU_NUM_REGISTER_LANE_WIDTH;
         i++)
    {
        gpuOut[i] = QPU_NUM_REGISTER_SENTINEL;
    }

    for (uint32_t launch = 0; launch < QPU_NUM_REGISTER_LAUNCHES; launch++)
    {
        /*
         * Physical uniform stream order per scheduler request:
         *   [0] output scratch row bus address
         *
         * Each request gets a distinct 16-word scratch row.  The kernel stores
         * one word from lane 0; the extra spacing prevents overlap if the VDW
         * setup is widened while this exploratory test is adjusted.
         */
        uint32_t *slot = gpuOut + launch * QPU_NUM_REGISTER_LANE_WIDTH;
        state->unif[launch][0] = GPU_BASE + (uint32_t)slot;
        state->unif_ptr[launch] = GPU_BASE + (uint32_t)&state->unif[launch];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             QPU_NUM_REGISTER_LAUNCHES);

    for (uint32_t launch = 0; launch < QPU_NUM_REGISTER_LAUNCHES; launch++)
    {
        out[launch] = gpuOut[launch * QPU_NUM_REGISTER_LANE_WIDTH];
    }

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
