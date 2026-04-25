#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "minimal_thrend_launch.h"
#include "minimal_threndshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 2

struct minimal_thrend_launch_state
{
    uint32_t code[sizeof(minimal_threndshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
};

int minimal_thrend_launch(struct vc4_runtime *rt)
{
    if (!rt || !rt->isInitialized)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    uint32_t handle = mem_alloc(sizeof(struct minimal_thrend_launch_state),
                                4096,
                                GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct minimal_thrend_launch_state *state =
        (volatile struct minimal_thrend_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, minimal_threndshader, sizeof state->code);

    for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] qpu_id
         *   [1] num_qpus
         *
         * The minimal qasm does not read these uniforms.  They are packed here
         * so the reference bundle exercises the ABI metadata contract without
         * adding any computation or memory-output behavior.
         */
        state->unif[qpu][0] = qpu;
        state->unif[qpu][1] = activeQpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             activeQpus);

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
