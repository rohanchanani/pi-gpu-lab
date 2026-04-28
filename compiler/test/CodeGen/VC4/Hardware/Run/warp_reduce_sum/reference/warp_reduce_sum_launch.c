#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "warp_reduce_sum_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 5u

struct warp_reduce_sum_gpu_state {
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[WARP_REDUCE_SUM_MAX_QPUS][NUM_UNIFS];
uint32_t unif_ptr[WARP_REDUCE_SUM_MAX_QPUS];
uint32_t handle;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t launch_count;
float payload[];
};

static uint32_t round_up_to_lane_width(uint32_t n)
{
const uint32_t lane_width = WARP_REDUCE_SUM_LANE_WIDTH;
if (n == 0)
return 0;
return (n + lane_width - 1u) & ~(lane_width - 1u);
}

static size_t launch_state_size(uint32_t padded_capacity_n)
{
return offsetof(struct warp_reduce_sum_gpu_state, payload) +
((size_t)2u * padded_capacity_n + WARP_REDUCE_SUM_GUARD_WORDS) *
sizeof(float);
}

static float *input_ptr(volatile struct warp_reduce_sum_gpu_state *gpu)
{
return (float *)gpu->payload;
}

static float *output_ptr(volatile struct warp_reduce_sum_gpu_state *gpu)
{
return (float *)gpu->payload + gpu->padded_capacity_n;
}

int warp_reduce_sum_prepare(
struct vc4_runtime *rt,
struct warp_reduce_sum_state *state,
uint32_t max_n)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (!rt->isInitialized) {
    if (vc4_runtime_init(rt) < 0)
        return -1;
}

uint32_t active_qpus = vc4_runtime_active_qpus(rt);
if (active_qpus != WARP_REDUCE_SUM_MAX_QPUS)
    return -1;
if (vc4_runtime_lane_width() != WARP_REDUCE_SUM_LANE_WIDTH)
    return -1;

uint32_t padded_capacity = round_up_to_lane_width(max_n);
if (padded_capacity < max_n)
    return -1;

size_t alloc_size = launch_state_size(padded_capacity);
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

volatile struct warp_reduce_sum_gpu_state *gpu =
    (volatile struct warp_reduce_sum_gpu_state *)(vc - GPU_BASE);

memset((void *)gpu, 0, alloc_size);
gpu->handle = handle;
gpu->max_n = max_n;
gpu->padded_capacity_n = padded_capacity;
gpu->launch_count = 0;

memcpy((void *)gpu->code, shader, sizeof gpu->code);

for (uint32_t qpu = 0; qpu < WARP_REDUCE_SUM_MAX_QPUS; qpu++)
    gpu->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpu->unif[qpu][0];

state->runtime = rt;
state->opaque = (void *)gpu;
state->prepared = 1;
state->max_n = max_n;
state->padded_capacity_n = padded_capacity;
state->runtime_allocations = 1;
state->runtime_launches = 0;

return 0;

}

int warp_reduce_sum_launch(
struct warp_reduce_sum_state *state,
const float *input,
float *out,
uint32_t n)
{
if (!state || !state->prepared || !state->opaque || !state->runtime)
return -1;
if (n != 0 && (!input || !out))
return -1;
if (n > state->max_n)
return -1;

volatile struct warp_reduce_sum_gpu_state *gpu =
    (volatile struct warp_reduce_sum_gpu_state *)state->opaque;

uint32_t active_qpus = vc4_runtime_active_qpus(state->runtime);
if (active_qpus != WARP_REDUCE_SUM_MAX_QPUS)
    return -1;

uint32_t padded_n = round_up_to_lane_width(n);
if (padded_n < n || padded_n > gpu->padded_capacity_n)
    return -1;

float *gpu_input = input_ptr(gpu);
float *gpu_output = output_ptr(gpu);

if (n != 0)
    memcpy(gpu_input, input, n * sizeof(float));

for (uint32_t i = n; i < padded_n; i++)
    gpu_input[i] = 0.0f;

for (uint32_t i = 0; i < gpu->padded_capacity_n + WARP_REDUCE_SUM_GUARD_WORDS; i++)
    gpu_output[i] = WARP_REDUCE_SUM_SENTINEL;

uint32_t gpu_input_addr = GPU_BASE + (uint32_t)gpu_input;
uint32_t gpu_output_addr = GPU_BASE + (uint32_t)gpu_output;

for (uint32_t qpu = 0; qpu < active_qpus; qpu++) {
    gpu->unif[qpu][0] = gpu_input_addr;
    gpu->unif[qpu][1] = gpu_output_addr;
    gpu->unif[qpu][2] = n;
    gpu->unif[qpu][3] = qpu;
    gpu->unif[qpu][4] = active_qpus;
    gpu->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpu->unif[qpu][0];
}

gpu_fft_base_exec_direct((uint32_t)gpu->code,
                         (uint32_t *)gpu->unif_ptr,
                         active_qpus);

if (n != 0)
    memcpy(out, gpu_output, n * sizeof(float));

for (uint32_t i = n; i < n + WARP_REDUCE_SUM_GUARD_WORDS &&
                     i < gpu->padded_capacity_n + WARP_REDUCE_SUM_GUARD_WORDS; i++) {
    if (gpu_output[i] != WARP_REDUCE_SUM_SENTINEL)
        return -1;
}

gpu->launch_count++;
state->runtime_launches = gpu->launch_count;

return 0;

}

void warp_reduce_sum_shutdown(struct warp_reduce_sum_state *state)
{
if (!state || !state->prepared || !state->opaque)
return;

volatile struct warp_reduce_sum_gpu_state *gpu =
    (volatile struct warp_reduce_sum_gpu_state *)state->opaque;
uint32_t handle = gpu->handle;

mem_unlock(handle);
mem_free(handle);

state->opaque = 0;
state->prepared = 0;

}

uint32_t warp_reduce_sum_runtime_allocations(
const struct warp_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_allocations;
}

uint32_t warp_reduce_sum_runtime_launches(
const struct warp_reduce_sum_state *state)
{
if (!state)
return 0;
return state->runtime_launches;
}

uint32_t warp_reduce_sum_runtime_capacity(
const struct warp_reduce_sum_state *state)
{
if (!state)
return 0;
return state->max_n;
}

