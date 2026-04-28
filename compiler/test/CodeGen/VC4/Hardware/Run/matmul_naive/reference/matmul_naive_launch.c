#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "matmul_naive_launch.h"
#include "matmul_naiveshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 8
#define B_GUARD_FLOATS 16u

struct matmul_naive_launch_state
{
    uint32_t code[sizeof(matmul_naiveshader) / sizeof(uint32_t)];
    uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
    uint32_t handle;
    uint32_t max_m;
    uint32_t max_n;
    uint32_t max_k;
    uint32_t launch_count;
    float payload[];
};

static volatile struct matmul_naive_launch_state *g_state;
static uint32_t g_handle;
static uint32_t g_allocations;

static size_t matmul_naive_count_a(uint32_t m, uint32_t k)
{
    return (size_t)m * (size_t)k;
}

static size_t matmul_naive_count_b(uint32_t n, uint32_t k)
{
    return (size_t)k * (size_t)n;
}

static size_t matmul_naive_count_c(uint32_t m, uint32_t n)
{
    return (size_t)m * (size_t)n;
}

static size_t matmul_naive_capacity_a(volatile struct matmul_naive_launch_state *state)
{
    return matmul_naive_count_a(state->max_m, state->max_k);
}

static size_t matmul_naive_capacity_b(volatile struct matmul_naive_launch_state *state)
{
    return matmul_naive_count_b(state->max_n, state->max_k) + B_GUARD_FLOATS;
}

static size_t matmul_naive_capacity_c(volatile struct matmul_naive_launch_state *state)
{
    return matmul_naive_count_c(state->max_m, state->max_n);
}

static size_t matmul_naive_state_size(uint32_t max_m, uint32_t max_n, uint32_t max_k)
{
    size_t aCapacity = (size_t)max_m * (size_t)max_k;
    size_t bCapacity = (size_t)max_k * (size_t)max_n + B_GUARD_FLOATS;
    size_t cCapacity = (size_t)max_m * (size_t)max_n;
    return offsetof(struct matmul_naive_launch_state, payload) +
           (aCapacity + bCapacity + cCapacity) * sizeof(float);
}

static float *matmul_naive_a_ptr(volatile struct matmul_naive_launch_state *state)
{
    return (float *)state->payload;
}

static float *matmul_naive_b_ptr(volatile struct matmul_naive_launch_state *state)
{
    return (float *)state->payload + matmul_naive_capacity_a(state);
}

static float *matmul_naive_c_ptr(volatile struct matmul_naive_launch_state *state)
{
    return (float *)state->payload +
           matmul_naive_capacity_a(state) +
           matmul_naive_capacity_b(state);
}

int matmul_naive_prepare(
    struct vc4_runtime *rt,
    uint32_t max_m,
    uint32_t max_n,
    uint32_t max_k)
{
    if (!rt || !rt->isInitialized)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    if (g_state)
    {
        /* This test/runtime shape is intentionally single-allocation. */
        if (max_m <= g_state->max_m &&
            max_n <= g_state->max_n &&
            max_k <= g_state->max_k)
            return 0;
        return -1;
    }

    size_t allocSize = matmul_naive_state_size(max_m, max_n, max_k);
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

    volatile struct matmul_naive_launch_state *state =
        (volatile struct matmul_naive_launch_state *)(vc - GPU_BASE);
    if (!state)
    {
        mem_unlock(handle);
        mem_free(handle);
        return -1;
    }

    memset((void *)state, 0, allocSize);
    state->handle = handle;
    state->max_m = max_m;
    state->max_n = max_n;
    state->max_k = max_k;
    state->launch_count = 0;

    /* Copy the assembled kernel code to GPU-visible memory exactly once. */
    memcpy((void *)state->code, matmul_naiveshader, sizeof state->code);

    for (uint32_t qpu = 0; qpu < VC4_RUNTIME_MAX_QPUS; qpu++)
        state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];

    g_handle = handle;
    g_state = state;
    g_allocations++;
    return 0;
}

int matmul_naive_launch(
    struct vc4_runtime *rt,
    const float *a,
    const float *b,
    float *c,
    uint32_t m,
    uint32_t n,
    uint32_t k)
{
    if (!rt || !rt->isInitialized)
        return -1;
    if (!g_state)
        return -1;
    if (m > g_state->max_m || n > g_state->max_n || k > g_state->max_k)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
        return -1;

    size_t aCount = matmul_naive_count_a(m, k);
    size_t bCount = matmul_naive_count_b(n, k);
    size_t cCount = matmul_naive_count_c(m, n);

    if ((aCount && !a) || (bCount && !b) || (cCount && !c))
        return -1;

    float *gpuA = matmul_naive_a_ptr(g_state);
    float *gpuB = matmul_naive_b_ptr(g_state);
    float *gpuC = matmul_naive_c_ptr(g_state);

    if (aCount)
        memcpy(gpuA, a, aCount * sizeof(float));
    if (bCount)
        memcpy(gpuB, b, bCount * sizeof(float));
    if (cCount)
        memcpy(gpuC, c, cCount * sizeof(float));

    /*
     * Pad only the private GPU B scratch region after this launch's compact
     * K*N payload.  Tail-lane TMU reads in the last B row may read up to 15
     * floats past logical B, but the qasm never stores those inactive lanes.
     */
    for (uint32_t i = 0; i < B_GUARD_FLOATS; i++)
        gpuB[bCount + i] = 0.0f;

    uint32_t gpuAAddr = GPU_BASE + (uint32_t)gpuA;
    uint32_t gpuBAddr = GPU_BASE + (uint32_t)gpuB;
    uint32_t gpuCAddr = GPU_BASE + (uint32_t)gpuC;

    for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
    {
        /*
         * Physical uniform stream order per QPU:
         *   [0] A scratch base address
         *   [1] B scratch base address
         *   [2] C scratch/result base address
         *   [3] M
         *   [4] N
         *   [5] K
         *   [6] qpu_id
         *   [7] num_qpus
         *
         * qpu_id and num_qpus are conceptual execution builtins in the MLIR
         * input even though this reference physically carries them as uniforms.
         */
        g_state->unif[qpu][0] = gpuAAddr;
        g_state->unif[qpu][1] = gpuBAddr;
        g_state->unif[qpu][2] = gpuCAddr;
        g_state->unif[qpu][3] = m;
        g_state->unif[qpu][4] = n;
        g_state->unif[qpu][5] = k;
        g_state->unif[qpu][6] = qpu;
        g_state->unif[qpu][7] = activeQpus;
        g_state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&g_state->unif[qpu];
    }

    gpu_fft_base_exec_direct((uint32_t)g_state->code,
                             (uint32_t *)g_state->unif_ptr,
                             activeQpus);

    if (cCount)
        memcpy(c, gpuC, cCount * sizeof(float));

    g_state->launch_count++;
    return 0;
}

void matmul_naive_release(struct vc4_runtime *rt)
{
    (void)rt;

    if (!g_state)
        return;

    /* One-time teardown only. The hardware test does not call this during the
     * shape loop; repeated kernel calls reuse the single allocation above. */
    mem_unlock(g_handle);
    mem_free(g_handle);
    g_state = 0;
    g_handle = 0;
}

uint32_t matmul_naive_runtime_allocations(void)
{
    return g_allocations;
}

uint32_t matmul_naive_runtime_launches(void)
{
    if (!g_state)
        return 0;
    return g_state->launch_count;
}

uint32_t matmul_naive_runtime_capacity_m(void)
{
    if (!g_state)
        return 0;
    return g_state->max_m;
}

uint32_t matmul_naive_runtime_capacity_n(void)
{
    if (!g_state)
        return 0;
    return g_state->max_n;
}

uint32_t matmul_naive_runtime_capacity_k(void)
{
    if (!g_state)
        return 0;
    return g_state->max_k;
}
