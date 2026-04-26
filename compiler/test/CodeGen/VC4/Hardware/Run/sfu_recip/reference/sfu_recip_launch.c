#include "rpi.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "sfu_recip_launch.h"
#include "sfu_recipshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 5

struct sfu_recip_launch_state
{
    uint32_t code[sizeof(sfu_recipshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    float payload[];
};

static uint32_t max_u32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

static size_t sfu_recip_state_size(uint32_t payload_elems)
{
    return offsetof(struct sfu_recip_launch_state, payload) +
           (size_t)2 * payload_elems * sizeof(float);
}

static float *sfu_recip_x_ptr(volatile struct sfu_recip_launch_state *state)
{
    return (float *)state->payload;
}

static float *sfu_recip_out_ptr(
    volatile struct sfu_recip_launch_state *state,
    uint32_t payload_elems)
{
    return (float *)state->payload + payload_elems;
}

int sfu_recip_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *out,
    uint32_t n)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!x || !out)
        return -1;

    uint32_t lane_width = vc4_runtime_lane_width();
    uint32_t active_qpus = vc4_runtime_active_qpus(rt);

    if (active_qpus == 0 || active_qpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    /*
     * Generated-style launchers should not reject semantic kernel arguments
     * such as n.  This kernel body processes one vector per active QPU, while
     * the harness supplies the exact size.  Allocate at least that many slots
     * to keep the hidden GPU-side payload safely padded even if a caller passes
     * a smaller n during manual debugging.
     */
    uint32_t required_elems = active_qpus * lane_width;
    uint32_t payload_elems = max_u32(n, required_elems);

    size_t alloc_size = sfu_recip_state_size(payload_elems);
    uint32_t handle = mem_alloc(alloc_size, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct sfu_recip_launch_state *state =
        (volatile struct sfu_recip_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, sfu_recipshader, sizeof state->code);

    float *gpu_x = sfu_recip_x_ptr(state);
    float *gpu_out = sfu_recip_out_ptr(state, payload_elems);

    memset(gpu_x, 0, (size_t)payload_elems * sizeof(float));
    memset(gpu_out, 0, (size_t)payload_elems * sizeof(float));
    memcpy(gpu_x, x, (size_t)n * sizeof(float));

    uint32_t gpu_x_addr = GPU_BASE + (uint32_t)gpu_x;
    uint32_t gpu_out_addr = GPU_BASE + (uint32_t)gpu_out;

    for (uint32_t qpu = 0; qpu < active_qpus; ++qpu)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] x base address
         *   [1] out base address
         *   [2] n element count
         *   [3] qpu_id
         *   [4] num_qpus
         *
         * qpu_id and num_qpus are conceptual execution builtins in the MLIR
         * input even though this reference physically carries them as uniforms.
         */
        state->unif[qpu][0] = gpu_x_addr;
        state->unif[qpu][1] = gpu_out_addr;
        state->unif[qpu][2] = n;
        state->unif[qpu][3] = qpu;
        state->unif[qpu][4] = active_qpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             active_qpus);

    memcpy(out, gpu_out, (size_t)n * sizeof(float));

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
