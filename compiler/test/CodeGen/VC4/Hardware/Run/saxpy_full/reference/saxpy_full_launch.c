#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "saxpy_full_launch.h"
#include "saxpy_fullshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 6

struct saxpy_full_launch_state
{
    uint32_t code[sizeof(saxpy_fullshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    uint32_t max_n;
    uint32_t padded_capacity_n;
    uint32_t launch_count;
    float payload[];
};

static volatile struct saxpy_full_launch_state *g_state;
static uint32_t g_handle;
static uint32_t g_allocations;

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

static uint32_t round_up_to_lane_width(uint32_t n)
{
    const uint32_t laneWidth = VC4_RUNTIME_LANE_WIDTH;
    if (n == 0)
        return 0;
    return (n + laneWidth - 1u) & ~(laneWidth - 1u);
}

static size_t saxpy_full_state_size(uint32_t padded_capacity_n)
{
    return offsetof(struct saxpy_full_launch_state, payload) +
           (size_t)2 * padded_capacity_n * sizeof(float);
}

static float *saxpy_full_x_ptr(volatile struct saxpy_full_launch_state *state)
{
    return (float *)state->payload;
}

static float *saxpy_full_y_ptr(volatile struct saxpy_full_launch_state *state)
{
    return (float *)state->payload + state->padded_capacity_n;
}

int saxpy_full_prepare(struct vc4_runtime *rt, uint32_t max_n)
{
    if (!rt || !rt->isInitialized)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    uint32_t paddedCapacity = round_up_to_lane_width(max_n);
    if (paddedCapacity < max_n)
        return -1;

    if (g_state)
    {
        /* This test/runtime shape is intentionally single-allocation. */
        if (max_n <= g_state->max_n)
            return 0;
        return -1;
    }

    size_t allocSize = saxpy_full_state_size(paddedCapacity);
    if (allocSize > 0xffffffffu)
        return -1;

    uint32_t handle = mem_alloc((uint32_t)allocSize, 4096, GPU_MEM_FLG);
    if (!handle)
        return -1;

    uint32_t vc = mem_lock(handle);
    if (!vc)
    {
        mem_free(handle);
        return -1;
    }

    volatile struct saxpy_full_launch_state *state =
        (volatile struct saxpy_full_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    memset((void *)state, 0, allocSize);
    state->handle = handle;
    state->max_n = max_n;
    state->padded_capacity_n = paddedCapacity;
    state->launch_count = 0;

    /* Copy the assembled kernel code to GPU-visible memory exactly once. */
    memcpy((void *)state->code, saxpy_fullshader, sizeof state->code);

    for (uint32_t qpu = 0; qpu < VC4_RUNTIME_MAX_QPUS; qpu++)
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];

    g_handle = handle;
    g_state = state;
    g_allocations++;
    return 0;
}

int saxpy_full_launch(
    struct vc4_runtime *rt,
    const float *x,
    float *y,
    float alpha,
    uint32_t n)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (n != 0 && (!x || !y))
        return -1;
    if (!g_state)
        return -1;
    if (n > g_state->max_n)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    uint32_t paddedN = round_up_to_lane_width(n);
    if (paddedN < n || paddedN > g_state->padded_capacity_n)
        return -1;

    float *gpuX = saxpy_full_x_ptr(g_state);
    float *gpuY = saxpy_full_y_ptr(g_state);

    if (n != 0)
    {
        memcpy(gpuX, x, n * sizeof(float));
        memcpy(gpuY, y, n * sizeof(float));
    }

    /*
     * Pad only the private GPU scratch buffers up to this launch's vector
     * boundary. The logical n uniform remains unchanged. Padding keeps
     * inactive tail-lane TMU reads inside the one allocated GPU buffer; the
     * qasm uses a dynamic VDW DEPTH so the final partial vector stores only
     * the logical tail count.
     */
    for (uint32_t i = n; i < paddedN; i++)
    {
        gpuX[i] = 0.0f;
        gpuY[i] = 0.0f;
    }

    uint32_t gpuXAddr = GPU_BASE + (uint32_t)gpuX;
    uint32_t gpuYAddr = GPU_BASE + (uint32_t)gpuY;

    for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] x scratch base address
         *   [1] y scratch base/result address
         *   [2] alpha
         *   [3] logical n, not paddedN
         *   [4] qpu_id
         *   [5] num_qpus
         *
         * qpu_id and num_qpus are conceptual execution builtins in the MLIR
         * input even though this reference physically carries them as uniforms.
         */
        g_state->unif[qpu][0] = gpuXAddr;
        g_state->unif[qpu][1] = gpuYAddr;
        g_state->unif[qpu][2] = float_as_u32(alpha);
        g_state->unif[qpu][3] = n;
        g_state->unif[qpu][4] = qpu;
        g_state->unif[qpu][5] = activeQpus;
        g_state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&g_state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)g_state->code,
                             (uint32_t *)g_state->unif_ptr,
                             activeQpus);

    if (n != 0)
        memcpy(y, gpuY, n * sizeof(float));

    g_state->launch_count++;
    return 0;
}

void saxpy_full_release(struct vc4_runtime *rt)
{
    (void)rt;

    if (!g_state)
        return;

    /* One-time teardown only. The hardware test does not call this during the
     * n-loop; repeated kernel calls reuse the single allocation above. */
    mem_unlock(g_handle);
    mem_free(g_handle);
    g_state = 0;
    g_handle = 0;
}

uint32_t saxpy_full_runtime_allocations(void)
{
    return g_allocations;
}

uint32_t saxpy_full_runtime_launches(void)
{
    if (!g_state)
        return 0;
    return g_state->launch_count;
}

uint32_t saxpy_full_runtime_capacity(void)
{
    if (!g_state)
        return 0;
    return g_state->max_n;
}
