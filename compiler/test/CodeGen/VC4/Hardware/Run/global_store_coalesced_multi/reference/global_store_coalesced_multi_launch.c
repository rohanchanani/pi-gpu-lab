#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "global_store_coalesced_multi_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 5u

struct global_store_coalesced_multi_gpu_state {
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS][NUM_UNIFS];
uint32_t unif_ptr[GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS];
uint32_t handle;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t launch_count;
uint32_t payload[];
};

static uint32_t round_up_to_lane_width(uint32_t n)
{
const uint32_t lane_width = GLOBAL_STORE_COALESCED_MULTI_LANE_WIDTH;
if (n == 0)
return 0;
return (n + lane_width - 1u) & ~(lane_width - 1u);
}

static size_t launch_state_size(uint32_t padded_capacity_n)
{
return offsetof(struct global_store_coalesced_multi_gpu_state, payload) +
((size_t)padded_capacity_n + GLOBAL_STORE_COALESCED_MULTI_GUARD_WORDS) *
sizeof(uint32_t);
}

static volatile uint32_t *payload_ptr(
volatile struct global_store_coalesced_multi_gpu_state *gpu)
{
return gpu->payload;
}

int global_store_coalesced_multi_prepare(
struct vc4_runtime *rt,
struct global_store_coalesced_multi_state *state,
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
if (active_qpus != GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS)
    return -1;
if (vc4_runtime_lane_width() != GLOBAL_STORE_COALESCED_MULTI_LANE_WIDTH)
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

volatile struct global_store_coalesced_multi_gpu_state *gpu =
    (volatile struct global_store_coalesced_multi_gpu_state *)(vc - GPU_BASE);

memset((void *)gpu, 0, alloc_size);
gpu->handle = handle;
gpu->max_n = max_n;
gpu->padded_capacity_n = padded_capacity;
gpu->launch_count = 0;

memcpy((void *)gpu->code, shader, sizeof gpu->code);

for (uint32_t qpu = 0; qpu < GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS; qpu++)
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

int global_store_coalesced_multi_launch(
struct global_store_coalesced_multi_state *state,
uint32_t *out,
uint32_t n,
uint32_t case_id)
{
if (!state || !state->prepared || !state->opaque || !state->runtime)
return -1;
if (n != 0 && !out)
return -1;
if (n > state->max_n)
return -1;

volatile struct global_store_coalesced_multi_gpu_state *gpu =
    (volatile struct global_store_coalesced_multi_gpu_state *)state->opaque;

uint32_t active_qpus = vc4_runtime_active_qpus(state->runtime);
if (active_qpus != GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS)
    return -1;

uint32_t padded_n = round_up_to_lane_width(n);
if (padded_n < n || padded_n > gpu->padded_capacity_n)
    return -1;

volatile uint32_t *gpu_out = payload_ptr(gpu);
uint32_t total_words = gpu->padded_capacity_n + GLOBAL_STORE_COALESCED_MULTI_GUARD_WORDS;

for (uint32_t i = 0; i < total_words; i++)
    gpu_out[i] = QPU_STORE_SENTINEL;

uint32_t gpu_out_addr = GPU_BASE + (uint32_t)gpu_out;

for (uint32_t qpu = 0; qpu < active_qpus; qpu++) {
    gpu->unif[qpu][0] = gpu_out_addr;
    gpu->unif[qpu][1] = n;
    gpu->unif[qpu][2] = case_id;
    gpu->unif[qpu][3] = qpu;
    gpu->unif[qpu][4] = active_qpus;
    gpu->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpu->unif[qpu][0];
}

gpu_fft_base_exec_direct((uint32_t)gpu->code,
                         (uint32_t *)gpu->unif_ptr,
                         active_qpus);

if (n != 0)
    memcpy(out, (const void *)gpu_out, n * sizeof(uint32_t));

for (uint32_t i = n; i < n + GLOBAL_STORE_COALESCED_MULTI_GUARD_WORDS &&
                     i < total_words; i++) {
    if (gpu_out[i] != QPU_STORE_SENTINEL)
        return -1;
}

gpu->launch_count++;
state->runtime_launches = gpu->launch_count;

return 0;

}

void global_store_coalesced_multi_shutdown(
struct global_store_coalesced_multi_state *state)
{
if (!state || !state->prepared || !state->opaque)
return;

volatile struct global_store_coalesced_multi_gpu_state *gpu =
    (volatile struct global_store_coalesced_multi_gpu_state *)state->opaque;
uint32_t handle = gpu->handle;

mem_unlock(handle);
mem_free(handle);

state->opaque = 0;
state->prepared = 0;

}

uint32_t global_store_coalesced_multi_runtime_allocations(
const struct global_store_coalesced_multi_state *state)
{
if (!state)
return 0;
return state->runtime_allocations;
}

uint32_t global_store_coalesced_multi_runtime_launches(
const struct global_store_coalesced_multi_state *state)
{
if (!state)
return 0;
return state->runtime_launches;
}

uint32_t global_store_coalesced_multi_runtime_capacity(
const struct global_store_coalesced_multi_state *state)
{
if (!state)
return 0;
return state->max_n;
}

