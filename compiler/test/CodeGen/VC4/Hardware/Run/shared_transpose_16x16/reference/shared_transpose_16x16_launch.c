#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "shared_transpose_16x16_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_L2CACTL (V3D_BASE + 0x00020)
#define V3D_SLCACTL (V3D_BASE + 0x00024)
#define V3D_SRQPC (V3D_BASE + 0x00430)
#define V3D_SRQUA (V3D_BASE + 0x00434)
#define V3D_SRQCS (V3D_BASE + 0x0043c)
#define V3D_VPMBASE (V3D_BASE + 0x00504)
#define V3D_DBCFG (V3D_BASE + 0x00e00)
#define V3D_DBQITE (V3D_BASE + 0x00e2c)
#define V3D_DBQITC (V3D_BASE + 0x00e30)
#define V3D_ERRSTAT (V3D_BASE + 0x00f20)

#define SHARED_TRANSPOSE_NUM_UNIFS 7u
#define SHARED_TRANSPOSE_TIMEOUT_USEC 2000000u
#define SHARED_TRANSPOSE_ERRSTAT_RELEVANT_MASK 0x0000efffu
#define SHARED_TRANSPOSE_VPM_URSV_4K 16u
#define SHARED_TRANSPOSE_SENTINEL 0x5aa55aa5u

struct shared_transpose_16x16_gpu_state
{
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK][SHARED_TRANSPOSE_NUM_UNIFS];
uint32_t unif_ptr[SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK];
uint32_t handle;
uint32_t input[SHARED_TRANSPOSE_16X16_WORDS];
uint32_t output[SHARED_TRANSPOSE_16X16_WORDS + SHARED_TRANSPOSE_16X16_GUARD_WORDS];
};

static volatile struct shared_transpose_16x16_gpu_state *g_gpu_state;
static uint32_t g_handle;
static uint32_t g_allocations;

static uint32_t gpu_addr(const volatile void *ptr)
{
return GPU_BASE + (uint32_t)ptr;
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
    if ((uint32_t)(now - start) > SHARED_TRANSPOSE_TIMEOUT_USEC)
        return -1;
}

return 0;

}

static int queue_kernel(struct shared_transpose_16x16_state *state)
{
clear_scheduler_and_caches();

uint32_t errBefore = GET32(V3D_ERRSTAT);

for (uint32_t warp = 0; warp < SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK; warp++)
{
    PUT32(V3D_SRQUA, g_gpu_state->unif_ptr[warp]);
    PUT32(V3D_SRQPC, (uint32_t)g_gpu_state->code);
}

int rc = wait_for_completions(SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK);
uint32_t errAfter = GET32(V3D_ERRSTAT);
state->srqcs_after_last_launch = GET32(V3D_SRQCS);

if (((errBefore ^ errAfter) & SHARED_TRANSPOSE_ERRSTAT_RELEVANT_MASK) != 0)
    state->errstat_relevant_changed++;

if (rc < 0)
{
    state->timeouts++;
    return -1;
}

return 0;

}

int shared_transpose_16x16_prepare(
struct vc4_runtime *rt,
struct shared_transpose_16x16_state *state)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (g_gpu_state)
    return -1;

if (vc4_runtime_init(rt) < 0)
    return -1;

uint32_t activeQpus = vc4_runtime_active_qpus(rt);
if (activeQpus < SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK)
    return -1;

PUT32(V3D_VPMBASE, SHARED_TRANSPOSE_VPM_URSV_4K);

uint32_t handle = mem_alloc(sizeof(struct shared_transpose_16x16_gpu_state),
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

volatile struct shared_transpose_16x16_gpu_state *gpuState =
    (volatile struct shared_transpose_16x16_gpu_state *)(vc - GPU_BASE);
memset((void *)gpuState, 0, sizeof *gpuState);
gpuState->handle = handle;

memcpy((void *)gpuState->code, shader, sizeof gpuState->code);

for (uint32_t warp = 0; warp < SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK; warp++)
    gpuState->unif_ptr[warp] = gpu_addr(&gpuState->unif[warp][0]);

g_gpu_state = gpuState;
g_handle = handle;
g_allocations++;

state->rt = rt;
state->prepared = 1;
state->active_qpus = activeQpus;
state->lane_width = vc4_runtime_lane_width();
state->warps_per_block = SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK;
state->runtime_allocations = g_allocations;
state->runtime_launches = 0;
state->timeouts = 0;
state->errstat_relevant_changed = 0;
state->vpmbase_readback = GET32(V3D_VPMBASE);
state->last_sentinel_mismatches = 0;
state->total_sentinel_mismatches = 0;

return 0;

}

int shared_transpose_16x16_launch(
struct shared_transpose_16x16_state *state,
const uint32_t input[SHARED_TRANSPOSE_16X16_WORDS],
uint32_t output[SHARED_TRANSPOSE_16X16_WORDS])
{
if (!state || !state->prepared || !g_gpu_state || !input || !output)
return -1;

memcpy((void *)g_gpu_state->input,
       input,
       SHARED_TRANSPOSE_16X16_WORDS * sizeof(uint32_t));

for (uint32_t i = 0;
     i < SHARED_TRANSPOSE_16X16_WORDS + SHARED_TRANSPOSE_16X16_GUARD_WORDS;
     i++)
{
    g_gpu_state->output[i] = SHARED_TRANSPOSE_SENTINEL;
}

uint32_t inputBus = gpu_addr(&g_gpu_state->input[0]);
uint32_t outputBus = gpu_addr(&g_gpu_state->output[0]);

for (uint32_t warp = 0; warp < SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK; warp++)
{
    g_gpu_state->unif[warp][0] = inputBus;
    g_gpu_state->unif[warp][1] = outputBus;
    g_gpu_state->unif[warp][2] = warp;
    g_gpu_state->unif[warp][3] = SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK;
    g_gpu_state->unif[warp][4] = 0u;
    g_gpu_state->unif[warp][5] = warp;
    g_gpu_state->unif[warp][6] = state->active_qpus;
    g_gpu_state->unif_ptr[warp] = gpu_addr(&g_gpu_state->unif[warp][0]);
}

if (queue_kernel(state) < 0)
    return -1;

uint32_t sentinelMismatches = 0;
for (uint32_t i = 0; i < SHARED_TRANSPOSE_16X16_GUARD_WORDS; i++)
{
    if (g_gpu_state->output[SHARED_TRANSPOSE_16X16_WORDS + i] !=
        SHARED_TRANSPOSE_SENTINEL)
    {
        sentinelMismatches++;
    }
}

state->last_sentinel_mismatches = sentinelMismatches;
state->total_sentinel_mismatches += sentinelMismatches;

memcpy(output,
       (const void *)g_gpu_state->output,
       SHARED_TRANSPOSE_16X16_WORDS * sizeof(uint32_t));

state->runtime_launches++;
return 0;

}

void shared_transpose_16x16_shutdown(
struct shared_transpose_16x16_state *state)
{
if (g_gpu_state)
{
mem_unlock(g_handle);
mem_free(g_handle);
g_gpu_state = 0;
g_handle = 0;
}

if (state && state->rt)
    vc4_runtime_shutdown(state->rt);

if (state)
    state->prepared = 0;

}

