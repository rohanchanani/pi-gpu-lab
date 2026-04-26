#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "saxpy_basic_launch.h"
#include "saxpy_basicshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 6

struct saxpy_basic_launch_state
{
    uint32_t code[sizeof(saxpy_basicshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    float payload[];
};

static uint32_t float_as_u32(float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

static size_t saxpy_basic_state_size(uint32_t n)
{
    return offsetof(struct saxpy_basic_launch_state, payload) +
           (size_t)2 * n * sizeof(float);
}

static float *saxpy_basic_x_ptr(
    volatile struct saxpy_basic_launch_state *state)
{
    return (float *)state->payload;
}

static float *saxpy_basic_y_ptr(
    volatile struct saxpy_basic_launch_state *state,
    uint32_t n)
{
    return (float *)state->payload + n;
}

int saxpy_basic_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *y,
    float alpha,
    uint32_t n)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!x || !y)
        return -1;

    uint32_t laneWidth = vc4_runtime_lane_width();
    uint32_t activeQpus = vc4_runtime_active_qpus(rt);

    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    /* One vector per active QPU for this small golden. */
    if (n != activeQpus * laneWidth)
        return -1;

    size_t allocSize = saxpy_basic_state_size(n);
    uint32_t handle = mem_alloc(allocSize, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct saxpy_basic_launch_state *state =
        (volatile struct saxpy_basic_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    state->handle = handle;
    memcpy((void *)state->code, saxpy_basicshader, sizeof state->code);

    float *gpuX = saxpy_basic_x_ptr(state);
    float *gpuY = saxpy_basic_y_ptr(state, n);
    memcpy(gpuX, x, n * sizeof(float));
    memcpy(gpuY, y, n * sizeof(float));

    uint32_t gpuXAddr = GPU_BASE + (uint32_t)gpuX;
    uint32_t gpuYAddr = GPU_BASE + (uint32_t)gpuY;

    for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] x
         *   [1] y
         *   [2] alpha
         *   [3] n
         *   [4] qpu_id
         *   [5] num_qpus
         *
         * qpu_id and num_qpus are conceptual execution builtins in the MLIR
         * input even though this reference physically carries them as uniforms.
         */
        state->unif[qpu][0] = gpuXAddr;
        state->unif[qpu][1] = gpuYAddr;
        state->unif[qpu][2] = float_as_u32(alpha);
        state->unif[qpu][3] = n;
        state->unif[qpu][4] = qpu;
        state->unif[qpu][5] = activeQpus;
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)state->code,
                             (uint32_t *)state->unif_ptr,
                             activeQpus);
    memcpy(y, gpuY, n * sizeof(float));

    mem_unlock(handle);
    mem_free(handle);
    return 0;
}
