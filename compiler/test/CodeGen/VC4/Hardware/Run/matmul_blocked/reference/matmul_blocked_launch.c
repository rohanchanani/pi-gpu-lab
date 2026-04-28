#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "matmul_blocked_launch.h"
#include "matmul_blockedshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_IDENT1 (V3D_BASE + 0x00004)
#define V3D_L2CACTL (V3D_BASE + 0x00020)
#define V3D_SLCACTL (V3D_BASE + 0x00024)
#define V3D_SQRSV0 (V3D_BASE + 0x00410)
#define V3D_SQRSV1 (V3D_BASE + 0x00414)
#define V3D_SRQPC (V3D_BASE + 0x00430)
#define V3D_SRQUA (V3D_BASE + 0x00434)
#define V3D_SRQCS (V3D_BASE + 0x0043c)
#define V3D_VPMBASE (V3D_BASE + 0x00504)
#define V3D_DBCFG (V3D_BASE + 0x00e00)
#define V3D_DBQITE (V3D_BASE + 0x00e2c)
#define V3D_DBQITC (V3D_BASE + 0x00e30)
#define V3D_ERRSTAT (V3D_BASE + 0x00f20)

#define MATMUL_BLOCKED_WARPS_PER_BLOCK 12u
#define MATMUL_BLOCKED_TILE_COLS 16u
#define MATMUL_BLOCKED_VPM_URSV_4K 16u
#define MATMUL_BLOCKED_NUM_UNIFS 14u
#define MATMUL_BLOCKED_TIMEOUT_USEC 2000000u
#define MATMUL_BLOCKED_ERRSTAT_RELEVANT_MASK 0x0000efffu

struct matmul_blocked_launch_state
{
    uint32_t code[sizeof(matmul_blockedshader) / sizeof(uint32_t)];
    uint32_t unif[MATMUL_BLOCKED_WARPS_PER_BLOCK][MATMUL_BLOCKED_NUM_UNIFS];
    uint32_t unif_ptr[MATMUL_BLOCKED_WARPS_PER_BLOCK];
    uint32_t handle;

    uint32_t max_m;
    uint32_t max_n;
    uint32_t max_k;
    uint32_t padded_m;
    uint32_t padded_n;
    uint32_t padded_k;

    uint32_t launch_count;
    uint32_t tile_wave_count;
    uint32_t timeout_count;
    uint32_t errstat_relevant_changed_count;
    uint32_t ident1;
    uint32_t vpmbase_readback;
    uint32_t srqcs_after_last_wave;

    float payload[];
};

static volatile struct matmul_blocked_launch_state *g_state;
static uint32_t g_handle;
static uint32_t g_allocations;

static uint32_t gpu_addr(const volatile void *ptr)
{
    return GPU_BASE + (uint32_t)ptr;
}

static uint32_t round_up(uint32_t value, uint32_t multiple)
{
    if (value == 0)
        return 0;
    return ((value + multiple - 1u) / multiple) * multiple;
}

static size_t matmul_blocked_state_size(
    uint32_t padded_m,
    uint32_t padded_n,
    uint32_t padded_k)
{
    size_t a_count = (size_t)padded_m * padded_k;
    size_t b_count = (size_t)padded_k * padded_n;
    size_t c_count = (size_t)padded_m * padded_n;
    return offsetof(struct matmul_blocked_launch_state, payload) +
           (a_count + b_count + c_count) * sizeof(float);
}

static float *matmul_blocked_a_ptr(volatile struct matmul_blocked_launch_state *state)
{
    return (float *)state->payload;
}

static float *matmul_blocked_b_ptr(volatile struct matmul_blocked_launch_state *state)
{
    return (float *)state->payload + (size_t)state->padded_m * state->padded_k;
}

static float *matmul_blocked_c_ptr(volatile struct matmul_blocked_launch_state *state)
{
    return matmul_blocked_b_ptr(state) + (size_t)state->padded_k * state->padded_n;
}

static void reserve_qpu_mask(uint32_t allowed_mask)
{
    uint32_t sqrsv0 = 0;
    uint32_t sqrsv1 = 0;

    for (uint32_t q = 0; q < 16u; q++)
    {
        uint32_t reserve_user_programs =
            ((allowed_mask & (1u << q)) == 0) ? 1u : 0u;

        if (q < 8u)
            sqrsv0 |= reserve_user_programs << (4u * q);
        else
            sqrsv1 |= reserve_user_programs << (4u * (q - 8u));
    }

    PUT32(V3D_SQRSV0, sqrsv0);
    PUT32(V3D_SQRSV1, sqrsv1);
}

static void clear_qpu_reservations(void)
{
    PUT32(V3D_SQRSV0, 0);
    PUT32(V3D_SQRSV1, 0);
}

static void clear_scheduler_and_caches(void)
{
    PUT32(V3D_DBCFG, 0);
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, 0xffffffffu);
    PUT32(V3D_L2CACTL, 1u << 2);
    PUT32(V3D_SLCACTL, 0xffffffffu);
    PUT32(V3D_SRQCS, (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16));
}

static int wait_for_completions(uint32_t expected)
{
    uint32_t start = (uint32_t)timer_get_usec();

    while ((((GET32(V3D_SRQCS) >> 16) & 0xffu) != expected))
    {
        uint32_t now = (uint32_t)timer_get_usec();
        if ((uint32_t)(now - start) > MATMUL_BLOCKED_TIMEOUT_USEC)
            return -1;
    }

    return 0;
}

static int launch_tile_wave(
    volatile struct matmul_blocked_launch_state *state,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t k_padded_limit,
    uint32_t tile_row_base,
    uint32_t tile_col_base)
{
    (void)m;
    (void)k;

    uint32_t gpuA = gpu_addr(matmul_blocked_a_ptr(state));
    uint32_t gpuB = gpu_addr(matmul_blocked_b_ptr(state));
    uint32_t gpuC = gpu_addr(matmul_blocked_c_ptr(state));

    for (uint32_t warp = 0; warp < MATMUL_BLOCKED_WARPS_PER_BLOCK; warp++)
    {
        state->unif[warp][0] = gpuA;
        state->unif[warp][1] = gpuB;
        state->unif[warp][2] = gpuC;
        state->unif[warp][3] = m;
        state->unif[warp][4] = n;
        state->unif[warp][5] = k;
        state->unif[warp][6] = state->padded_k;
        state->unif[warp][7] = state->padded_n;
        state->unif[warp][8] = tile_row_base;
        state->unif[warp][9] = tile_col_base;
        state->unif[warp][10] = warp;
        state->unif[warp][11] = MATMUL_BLOCKED_WARPS_PER_BLOCK;
        state->unif[warp][12] = k_padded_limit;
        state->unif[warp][13] = 0u;
        state->unif_ptr[warp] = gpu_addr(&state->unif[warp][0]);
    }

    reserve_qpu_mask(0x0fffu);
    clear_scheduler_and_caches();

    uint32_t err_before = GET32(V3D_ERRSTAT);

    for (uint32_t warp = 0; warp < MATMUL_BLOCKED_WARPS_PER_BLOCK; warp++)
    {
        PUT32(V3D_SRQUA, state->unif_ptr[warp]);
        PUT32(V3D_SRQPC, (uint32_t)state->code);
    }

    int rc = wait_for_completions(MATMUL_BLOCKED_WARPS_PER_BLOCK);
    uint32_t err_after = GET32(V3D_ERRSTAT);
    state->srqcs_after_last_wave = GET32(V3D_SRQCS);
    clear_qpu_reservations();

    state->tile_wave_count++;

    if (((err_before ^ err_after) & MATMUL_BLOCKED_ERRSTAT_RELEVANT_MASK) != 0)
        state->errstat_relevant_changed_count++;

    if (rc < 0)
    {
        state->timeout_count++;
        return -1;
    }

    return 0;
}

int matmul_blocked_prepare(
    struct vc4_runtime *rt,
    uint32_t max_m,
    uint32_t max_n,
    uint32_t max_k)
{
    if (!rt || !rt->isInitialized)
        return -1;

    uint32_t activeQpus = vc4_runtime_active_qpus(rt);
    if (activeQpus != MATMUL_BLOCKED_WARPS_PER_BLOCK)
        return -1;

    if (g_state)
    {
        if (max_m <= g_state->max_m &&
            max_n <= g_state->max_n &&
            max_k <= g_state->max_k)
            return 0;
        return -1;
    }

    uint32_t paddedM = round_up(max_m, MATMUL_BLOCKED_WARPS_PER_BLOCK);
    uint32_t paddedN = round_up(max_n, MATMUL_BLOCKED_TILE_COLS);
    uint32_t paddedK = round_up(max_k, MATMUL_BLOCKED_WARPS_PER_BLOCK);
    if (paddedM < max_m || paddedN < max_n || paddedK < max_k)
        return -1;

    size_t allocSize = matmul_blocked_state_size(paddedM, paddedN, paddedK);
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

    volatile struct matmul_blocked_launch_state *state =
        (volatile struct matmul_blocked_launch_state *)(vc - GPU_BASE);
    memset((void *)state, 0, allocSize);

    state->handle = handle;
    state->max_m = max_m;
    state->max_n = max_n;
    state->max_k = max_k;
    state->padded_m = paddedM;
    state->padded_n = paddedN;
    state->padded_k = paddedK;
    state->ident1 = GET32(V3D_IDENT1);

    PUT32(V3D_VPMBASE, MATMUL_BLOCKED_VPM_URSV_4K);
    state->vpmbase_readback = GET32(V3D_VPMBASE);

    memcpy((void *)state->code, matmul_blockedshader, sizeof state->code);
    for (uint32_t warp = 0; warp < MATMUL_BLOCKED_WARPS_PER_BLOCK; warp++)
        state->unif_ptr[warp] = gpu_addr(&state->unif[warp][0]);

    g_handle = handle;
    g_state = state;
    g_allocations++;
    return 0;
}

int matmul_blocked_launch(
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
    if ((m != 0 && k != 0 && !a) || (k != 0 && n != 0 && !b))
        return -1;
    if (m != 0 && n != 0 && !c)
        return -1;

    float *gpuA = matmul_blocked_a_ptr(g_state);
    float *gpuB = matmul_blocked_b_ptr(g_state);
    float *gpuC = matmul_blocked_c_ptr(g_state);

    size_t a_count = (size_t)g_state->padded_m * g_state->padded_k;
    size_t b_count = (size_t)g_state->padded_k * g_state->padded_n;
    size_t c_count = (size_t)g_state->padded_m * g_state->padded_n;
    memset(gpuA, 0, a_count * sizeof(float));
    memset(gpuB, 0, b_count * sizeof(float));
    memset(gpuC, 0, c_count * sizeof(float));

    for (uint32_t row = 0; row < m; row++)
    {
        if (k != 0)
            memcpy(gpuA + (size_t)row * g_state->padded_k,
                   a + (size_t)row * k,
                   (size_t)k * sizeof(float));
    }

    for (uint32_t row = 0; row < k; row++)
    {
        if (n != 0)
            memcpy(gpuB + (size_t)row * g_state->padded_n,
                   b + (size_t)row * n,
                   (size_t)n * sizeof(float));
    }

    uint32_t kPaddedLimit = round_up(k, MATMUL_BLOCKED_WARPS_PER_BLOCK);
    int rc = 0;

    if (m != 0 && n != 0)
    {
        for (uint32_t tileRow = 0; tileRow < m; tileRow += MATMUL_BLOCKED_WARPS_PER_BLOCK)
        {
            for (uint32_t tileCol = 0; tileCol < n; tileCol += MATMUL_BLOCKED_TILE_COLS)
            {
                if (launch_tile_wave(g_state, m, n, k, kPaddedLimit, tileRow, tileCol) < 0)
                {
                    rc = -1;
                    goto done;
                }
            }
        }
    }

done:
    if (rc == 0 && m != 0 && n != 0)
    {
        for (uint32_t row = 0; row < m; row++)
        {
            memcpy(c + (size_t)row * n,
                   gpuC + (size_t)row * g_state->padded_n,
                   (size_t)n * sizeof(float));
        }
    }

    g_state->launch_count++;
    return rc;
}

void matmul_blocked_release(struct vc4_runtime *rt)
{
    (void)rt;
    if (!g_state)
        return;

    mem_unlock(g_handle);
    mem_free(g_handle);
    g_state = 0;
    g_handle = 0;
}

uint32_t matmul_blocked_runtime_allocations(void)
{
    return g_allocations;
}

uint32_t matmul_blocked_runtime_launches(void)
{
    if (!g_state)
        return 0;
    return g_state->launch_count;
}

uint32_t matmul_blocked_runtime_tile_waves(void)
{
    if (!g_state)
        return 0;
    return g_state->tile_wave_count;
}

uint32_t matmul_blocked_runtime_timeouts(void)
{
    if (!g_state)
        return 0;
    return g_state->timeout_count;
}

uint32_t matmul_blocked_runtime_errstat_relevant_changed(void)
{
    if (!g_state)
        return 0;
    return g_state->errstat_relevant_changed_count;
}

uint32_t matmul_blocked_runtime_vpmbase_readback(void)
{
    if (!g_state)
        return 0;
    return g_state->vpmbase_readback;
}

uint32_t matmul_blocked_runtime_ident1(void)
{
    if (!g_state)
        return 0;
    return g_state->ident1;
}

uint32_t matmul_blocked_runtime_srqcs_after_last_wave(void)
{
    if (!g_state)
        return 0;
    return g_state->srqcs_after_last_wave;
}

uint32_t matmul_blocked_runtime_capacity_m(void)
{
    if (!g_state)
        return 0;
    return g_state->max_m;
}

uint32_t matmul_blocked_runtime_capacity_n(void)
{
    if (!g_state)
        return 0;
    return g_state->max_n;
}

uint32_t matmul_blocked_runtime_capacity_k(void)
{
    if (!g_state)
        return 0;
    return g_state->max_k;
}
