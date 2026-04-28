#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "block_reduce_sum_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_IDENT1 (V3D_BASE + 0x00004)
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

#define NUM_UNIFS 6u
#define BLOCK_REDUCE_SUM_TIMEOUT_USEC 2000000u
#define BLOCK_REDUCE_SUM_ERRSTAT_RELEVANT_MASK 0x0000efffu

struct block_reduce_sum_gpu_state {
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[BLOCK_REDUCE_SUM_MAX_WARPS_PER_BLOCK][NUM_UNIFS];
uint32_t unif_ptr[BLOCK_REDUCE_SUM_MAX_WARPS_PER_BLOCK];
uint32_t handle;
uint32_t max_blocks;
uint32_t max_values_per_block;
uint32_t launch_count;
uint32_t tile_wave_count;
uint32_t timeout_count;
uint32_t errstat_relevant_changed_count;
uint32_t ident1;
uint32_t vpmbase_readback;
uint32_t srqcs_after_last_wave;
float payload[];
};

static float *input_ptr(volatile struct block_reduce_sum_gpu_state *gpu)
{
return (float *)gpu->payload;
}

static float *output_ptr(volatile struct block_reduce_sum_gpu_state *gpu)
{
return (float *)gpu->payload +
(size_t)gpu->max_blocks * gpu->max_values_per_block;
}

static size_t launch_state_size(uint32_t max_blocks,
uint32_t max_values_per_block)
{
size_t input_count = (size_t)max_blocks * max_values_per_block;
size_t output_count = (size_t)max_blocks * BLOCK_REDUCE_SUM_RESULT_LANES;
return offsetof(struct block_reduce_sum_gpu_state, payload) +
(input_count + output_count) * sizeof(float);
}

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

while ((((GET32(V3D_SRQCS) >> 16) & 0xffu) != expected)) {
    uint32_t now = (uint32_t)timer_get_usec();
    if ((uint32_t)(now - start) > BLOCK_REDUCE_SUM_TIMEOUT_USEC)
        return -1;
}

return 0;

}

static int launch_block_wave(
volatile struct block_reduce_sum_gpu_state *gpu,
uint32_t values_per_block,
uint32_t block_index,
uint32_t warps_per_block)
{
float *base_input = input_ptr(gpu);
float *base_output = output_ptr(gpu);
uint32_t block_input_addr =
gpu_addr(base_input + (size_t)block_index * gpu->max_values_per_block);
uint32_t block_output_addr =
gpu_addr(base_output + (size_t)block_index * BLOCK_REDUCE_SUM_RESULT_LANES);

for (uint32_t warp = 0; warp < warps_per_block; warp++) {
    gpu->unif[warp][0] = block_input_addr;
    gpu->unif[warp][1] = block_output_addr;
    gpu->unif[warp][2] = values_per_block;
    gpu->unif[warp][3] = warp;
    gpu->unif[warp][4] = warps_per_block;
    gpu->unif[warp][5] = 0u;
    gpu->unif_ptr[warp] = gpu_addr(&gpu->unif[warp][0]);
}

clear_scheduler_and_caches();

uint32_t err_before = GET32(V3D_ERRSTAT);

for (uint32_t warp = 0; warp < warps_per_block; warp++) {
    PUT32(V3D_SRQUA, gpu->unif_ptr[warp]);
    PUT32(V3D_SRQPC, (uint32_t)gpu->code);
}

int rc = wait_for_completions(warps_per_block);
uint32_t err_after = GET32(V3D_ERRSTAT);
gpu->srqcs_after_last_wave = GET32(V3D_SRQCS);
gpu->tile_wave_count++;

if (((err_before ^ err_after) & BLOCK_REDUCE_SUM_ERRSTAT_RELEVANT_MASK) != 0)
    gpu->errstat_relevant_changed_count++;

if (rc < 0) {
    gpu->timeout_count++;
    return -1;
}

return 0;

}

int block_reduce_sum_prepare(
struct vc4_runtime *rt,
struct block_reduce_sum_state *state,
uint32_t max_blocks,
uint32_t max_values_per_block)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (!rt->isInitialized) {
    if (vc4_runtime_init(rt) < 0)
        return -1;
}

if (vc4_runtime_active_qpus(rt) != BLOCK_REDUCE_SUM_MAX_QPUS)
    return -1;
if (vc4_runtime_lane_width() != BLOCK_REDUCE_SUM_LANE_WIDTH)
    return -1;
if (max_blocks == 0 || max_values_per_block == 0)
    return -1;

size_t alloc_size = launch_state_size(max_blocks, max_values_per_block);
if (alloc_size > 0xffffffffu)
    return -1;

uint32_t handle = mem_alloc((uint32_t)alloc_size, 4096, GPU_MEM_FLG);
if (!handle)
    return -1;

uint32_t vc = mem_lock(handle);
if (!vc) {
    mem_free(handle);
    return -1;
}

volatile struct block_reduce_sum_gpu_state *gpu =
    (volatile struct block_reduce_sum_gpu_state *)(vc - GPU_BASE);

memset((void *)gpu, 0, alloc_size);
gpu->handle = handle;
gpu->max_blocks = max_blocks;
gpu->max_values_per_block = max_values_per_block;
gpu->ident1 = GET32(V3D_IDENT1);

PUT32(V3D_VPMBASE, BLOCK_REDUCE_SUM_VPM_URSV_4K);
gpu->vpmbase_readback = GET32(V3D_VPMBASE);

memcpy((void *)gpu->code, shader, sizeof gpu->code);

for (uint32_t warp = 0; warp < BLOCK_REDUCE_SUM_MAX_WARPS_PER_BLOCK; warp++)
    gpu->unif_ptr[warp] = gpu_addr(&gpu->unif[warp][0]);

state->runtime = rt;
state->opaque = (void *)gpu;
state->prepared = 1;
state->max_blocks = max_blocks;
state->max_values_per_block = max_values_per_block;
state->runtime_allocations = 1;
state->runtime_ident1 = gpu->ident1;
state->runtime_vpmbase_readback = gpu->vpmbase_readback;

return 0;

}

int block_reduce_sum_launch(
struct block_reduce_sum_state *state,
const float *input,
float *out,
uint32_t blocks,
uint32_t values_per_block,
uint32_t warps_per_block)
{
if (!state || !state->prepared || !state->opaque || !state->runtime)
return -1;
if (blocks == 0 || blocks > state->max_blocks)
return -1;
if (values_per_block == 0 ||
values_per_block > state->max_values_per_block)
return -1;
if (warps_per_block == 0 ||
warps_per_block > BLOCK_REDUCE_SUM_MAX_WARPS_PER_BLOCK)
return -1;
if (values_per_block > warps_per_block * BLOCK_REDUCE_SUM_LANE_WIDTH)
return -1;
if (!input || !out)
return -1;

volatile struct block_reduce_sum_gpu_state *gpu =
    (volatile struct block_reduce_sum_gpu_state *)state->opaque;

float *gpu_input = input_ptr(gpu);
float *gpu_output = output_ptr(gpu);

for (uint32_t block = 0; block < blocks; block++) {
    float *dst = gpu_input + (size_t)block * gpu->max_values_per_block;
    const float *src = input + (size_t)block * values_per_block;

    memcpy(dst, src, values_per_block * sizeof(float));
    for (uint32_t i = values_per_block; i < gpu->max_values_per_block; i++)
        dst[i] = 0.0f;
}

for (uint32_t i = 0; i < blocks * BLOCK_REDUCE_SUM_RESULT_LANES; i++)
    gpu_output[i] = BLOCK_REDUCE_SUM_SENTINEL;

int rc = 0;
for (uint32_t block = 0; block < blocks; block++) {
    if (launch_block_wave(gpu,
                          values_per_block,
                          block,
                          warps_per_block) < 0) {
        rc = -1;
        break;
    }
}

if (rc == 0) {
    memcpy(out,
           gpu_output,
           (size_t)blocks * BLOCK_REDUCE_SUM_RESULT_LANES * sizeof(float));
}

gpu->launch_count++;

state->runtime_launches = gpu->launch_count;
state->runtime_tile_waves = gpu->tile_wave_count;
state->runtime_timeouts = gpu->timeout_count;
state->runtime_errstat_relevant_changed =
    gpu->errstat_relevant_changed_count;
state->runtime_srqcs_after_last_wave = gpu->srqcs_after_last_wave;

return rc;

}

void block_reduce_sum_shutdown(struct block_reduce_sum_state *state)
{
if (!state || !state->prepared || !state->opaque)
return;

volatile struct block_reduce_sum_gpu_state *gpu =
    (volatile struct block_reduce_sum_gpu_state *)state->opaque;
uint32_t handle = gpu->handle;

mem_unlock(handle);
mem_free(handle);

state->opaque = 0;
state->prepared = 0;

}

uint32_t block_reduce_sum_runtime_allocations(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_allocations;
}

uint32_t block_reduce_sum_runtime_launches(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_launches;
}

uint32_t block_reduce_sum_runtime_tile_waves(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_tile_waves;
}

uint32_t block_reduce_sum_runtime_timeouts(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_timeouts;
}

uint32_t block_reduce_sum_runtime_errstat_relevant_changed(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_errstat_relevant_changed;
}

uint32_t block_reduce_sum_runtime_capacity_blocks(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->max_blocks;
}

uint32_t block_reduce_sum_runtime_capacity_values_per_block(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->max_values_per_block;
}

uint32_t block_reduce_sum_runtime_vpmbase_readback(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_vpmbase_readback;
}

uint32_t block_reduce_sum_runtime_ident1(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_ident1;
}

uint32_t block_reduce_sum_runtime_srqcs_after_last_wave(
const struct block_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_srqcs_after_last_wave;
}

