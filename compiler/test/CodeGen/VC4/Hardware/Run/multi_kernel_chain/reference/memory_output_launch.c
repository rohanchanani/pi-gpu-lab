#include "rpi.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "memory_output_launch.h"
#include "memory_outputshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 3

struct memory_output_launch_state
{
    uint32_t code[sizeof(memory_outputshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
};

static size_t align_up_size(size_t value, size_t align)
{
    return (value + align - 1) & ~(align - 1);
}

static uint32_t active_output_words(uint32_t active_qpus)
{
    return active_qpus * MEMORY_OUTPUT_WORDS_PER_QPU;
}

static size_t memory_output_payload_offset(void)
{
    return align_up_size(sizeof(struct memory_output_launch_state), 16);
}

static size_t memory_output_launch_state_size(uint32_t active_qpus)
{
    return memory_output_payload_offset() +
           (size_t)active_output_words(active_qpus) * sizeof(uint32_t);
}

static uint32_t *memory_output_payload_ptr(
    volatile struct memory_output_launch_state *state)
{
    return (uint32_t *)((uint8_t *)state + memory_output_payload_offset());
}

int memory_output_launch(struct vc4_runtime *rt,
                         uint32_t *out,
                         uint32_t out_words)
{
    uint32_t active_qpus = vc4_runtime_active_qpus(rt);
    uint32_t required_words = active_output_words(active_qpus);

    if (!rt || !rt->isInitialized)
        return -1;
    if (!out)
        return -1;
    if (active_qpus == 0 || active_qpus > VC4_RUNTIME_MAX_QPUS)
        return -1;
    if (out_words < required_words)
        return -1;

    size_t alloc_size = memory_output_launch_state_size(active_qpus);
    uint32_t handle = mem_alloc(alloc_size, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct memory_output_launch_state *state =
        (volatile struct memory_output_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, memory_outputshader, sizeof state->code);

    uint32_t *gpu_output = memory_output_payload_ptr(state);
    memset(gpu_output, 0xA5, (size_t)required_words * sizeof(uint32_t));

    uint32_t gpu_output_addr = GPU_BASE + (uint32_t)gpu_output;

    for (uint32_t qpu = 0; qpu < active_qpus; qpu++)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] output base bus address
         *   [1] qpu_id
         *   [2] num_qpus
         *
         * qpu_id and num_qpus are execution builtins conceptually.  They are
         * physically carried in the uniform stream suffix for this reference
         * bundle.
         */
        state->unif[qpu][0] = gpu_output_addr;
        state->unif[qpu][1] = qpu;
        state->unif[qpu][2] = active_qpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             active_qpus);

    memcpy(out, gpu_output, (size_t)required_words * sizeof(uint32_t));

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
