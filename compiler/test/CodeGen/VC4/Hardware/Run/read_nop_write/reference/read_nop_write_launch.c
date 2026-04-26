#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "read_nop_write_launch.h"
#include "read_nop_writeshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 5

struct read_nop_write_launch_state
{
    uint32_t code[sizeof(read_nop_writeshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    uint32_t payload[];
};

static size_t read_nop_write_launch_state_size(uint32_t n)
{
    return offsetof(struct read_nop_write_launch_state, payload) +
           (size_t)2 * n * sizeof(uint32_t);
}

static uint32_t *read_nop_write_input_ptr(
    volatile struct read_nop_write_launch_state *state)
{
    return (uint32_t *)state->payload;
}

static uint32_t *read_nop_write_result_ptr(
    volatile struct read_nop_write_launch_state *state,
    uint32_t n)
{
    return (uint32_t *)state->payload + n;
}

int read_nop_write_launch(
    struct vc4_runtime *rt,
    const uint32_t *input,
    uint32_t *result,
    uint32_t n)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!input || !result)
        return -1;

    uint32_t lane_width = vc4_runtime_lane_width();
    uint32_t active_qpus = vc4_runtime_active_qpus(rt);

    if (active_qpus == 0 || active_qpus > VC4_RUNTIME_MAX_QPUS)
        return -1;
    if (n != active_qpus * lane_width)
        return -1;

    size_t alloc_size = read_nop_write_launch_state_size(n);
    uint32_t handle = mem_alloc(alloc_size, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct read_nop_write_launch_state *state =
        (volatile struct read_nop_write_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, read_nop_writeshader, sizeof state->code);

    uint32_t *gpu_input = read_nop_write_input_ptr(state);
    uint32_t *gpu_result = read_nop_write_result_ptr(state, n);

    memcpy(gpu_input, input, (size_t)n * sizeof(uint32_t));
    memset(gpu_result, 0, (size_t)n * sizeof(uint32_t));

    uint32_t gpu_input_addr = GPU_BASE + (uint32_t)gpu_input;
    uint32_t gpu_result_addr = GPU_BASE + (uint32_t)gpu_result;

    for (uint32_t qpu = 0; qpu < active_qpus; qpu++)
    {
        /*
         * Physical uniform stream order per active QPU:
         *   [0] input base address
         *   [1] result/output base address
         *   [2] n word count
         *   [3] qpu_id
         *   [4] num_qpus
         *
         * qpu_id and num_qpus are conceptual execution builtins even though
         * this reference implementation materializes them through the uniform
         * suffix.
         */
        state->unif[qpu][0] = gpu_input_addr;
        state->unif[qpu][1] = gpu_result_addr;
        state->unif[qpu][2] = n;
        state->unif[qpu][3] = qpu;
        state->unif[qpu][4] = active_qpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct(
        (uint32_t)state->code,
        (uint32_t *)state->unif_ptr,
        active_qpus);

    memcpy(result, gpu_result, (size_t)n * sizeof(uint32_t));

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
