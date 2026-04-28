#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "tmu_strided_load_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_UNIFS 9u

struct tmu_strided_load_gpu_state {
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[TMU_STRIDED_LOAD_MAX_QPUS][NUM_UNIFS];
uint32_t unif_ptr[TMU_STRIDED_LOAD_MAX_QPUS];
uint32_t handle;
uint32_t max_input_words;
uint32_t max_n;
uint32_t padded_output_capacity_n;
uint32_t launch_count;
float payload[];
};

static uint32_t float_as_u32(float value)
{
union {
float f;
uint32_t u;
} bits;
bits.f = value;
return bits.u;
}

static uint32_t round_up_to_lane_width(uint32_t n)
{
const uint32_t lane_width = TMU_STRIDED_LOAD_LANE_WIDTH;
if (n == 0)
return 0;
return (n + lane_width - 1u) & ~(lane_width - 1u);
}

static size_t launch_state_size(uint32_t max_input_words,
uint32_t padded_output_capacity_n)
{
return offsetof(struct tmu_strided_load_gpu_state, payload) +
((size_t)max_input_words +
(size_t)padded_output_capacity_n +
TMU_STRIDED_LOAD_GUARD_WORDS) *
sizeof(float);
}

static float *input_ptr(volatile struct tmu_strided_load_gpu_state *gpu)
{
return (float *)gpu->payload;
}

static float *output_ptr(volatile struct tmu_strided_load_gpu_state *gpu)
{
return (float *)gpu->payload + gpu->max_input_words;
}

int tmu_strided_load_prepare(
struct vc4_runtime *rt,
struct tmu_strided_load_state *state,
uint32_t max_input_words,
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
if (active_qpus != TMU_STRIDED_LOAD_MAX_QPUS)
    return -1;
if (vc4_runtime_lane_width() != TMU_STRIDED_LOAD_LANE_WIDTH)
    return -1;

uint32_t padded_output_capacity = round_up_to_lane_width(max_n);
if (padded_output_capacity < max_n)
    return -1;

size_t alloc_size = launch_state_size(max_input_words,
                                      padded_output_capacity);
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

volatile struct tmu_strided_load_gpu_state *gpu =
    (volatile struct tmu_strided_load_gpu_state *)(vc - GPU_BASE);

memset((void *)gpu, 0, alloc_size);
gpu->handle = handle;
gpu->max_input_words = max_input_words;
gpu->max_n = max_n;
gpu->padded_output_capacity_n = padded_output_capacity;
gpu->launch_count = 0;

memcpy((void *)gpu->code, shader, sizeof gpu->code);

for (uint32_t qpu = 0; qpu < TMU_STRIDED_LOAD_MAX_QPUS; qpu++)
    gpu->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpu->unif[qpu][0];

state->runtime = rt;
state->opaque = (void *)gpu;
state->prepared = 1;
state->max_input_words = max_input_words;
state->max_n = max_n;
state->padded_output_capacity_n = padded_output_capacity;
state->runtime_allocations = 1;
state->runtime_launches = 0;

return 0;

}

int tmu_strided_load_launch(
struct tmu_strided_load_state *state,
const float *input,
float *out,
uint32_t n,
uint32_t offset,
uint32_t stride,
float scale,
float bias)
{
if (!state || !state->prepared || !state->opaque || !state->runtime)
return -1;
if (!input)
return -1;
if (n != 0 && !out)
return -1;
if (n > state->max_n)
return -1;
if (stride == 0)
return -1;

volatile struct tmu_strided_load_gpu_state *gpu =
    (volatile struct tmu_strided_load_gpu_state *)state->opaque;

uint32_t active_qpus = vc4_runtime_active_qpus(state->runtime);
if (active_qpus != TMU_STRIDED_LOAD_MAX_QPUS)
    return -1;

uint32_t padded_n = round_up_to_lane_width(n);
uint32_t max_source_index = offset;
if (padded_n != 0)
    max_source_index = offset + (padded_n - 1u) * stride;
if (max_source_index >= state->max_input_words)
    return -1;

float *gpu_input = input_ptr(gpu);
float *gpu_output = output_ptr(gpu);

memcpy(gpu_input, input, state->max_input_words * sizeof(float));

for (uint32_t i = 0; i < gpu->padded_output_capacity_n + TMU_STRIDED_LOAD_GUARD_WORDS; i++)
    gpu_output[i] = TMU_STRIDED_LOAD_SENTINEL;

uint32_t gpu_input_addr = GPU_BASE + (uint32_t)gpu_input;
uint32_t gpu_output_addr = GPU_BASE + (uint32_t)gpu_output;

for (uint32_t qpu = 0; qpu < active_qpus; qpu++) {
    gpu->unif[qpu][0] = gpu_input_addr;
    gpu->unif[qpu][1] = gpu_output_addr;
    gpu->unif[qpu][2] = n;
    gpu->unif[qpu][3] = offset;
    gpu->unif[qpu][4] = stride;
    gpu->unif[qpu][5] = float_as_u32(scale);
    gpu->unif[qpu][6] = float_as_u32(bias);
    gpu->unif[qpu][7] = qpu;
    gpu->unif[qpu][8] = active_qpus;
    gpu->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpu->unif[qpu][0];
}

gpu_fft_base_exec_direct((uint32_t)gpu->code,
                         (uint32_t *)gpu->unif_ptr,
                         active_qpus);

if (n != 0)
    memcpy(out, gpu_output, n * sizeof(float));

for (uint32_t i = n; i < n + TMU_STRIDED_LOAD_GUARD_WORDS &&
                     i < gpu->padded_output_capacity_n + TMU_STRIDED_LOAD_GUARD_WORDS; i++) {
    if (gpu_output[i] != TMU_STRIDED_LOAD_SENTINEL)
        return -1;
}

gpu->launch_count++;
state->runtime_launches = gpu->launch_count;

return 0;

}

void tmu_strided_load_shutdown(struct tmu_strided_load_state *state)
{
if (!state || !state->prepared || !state->opaque)
return;

volatile struct tmu_strided_load_gpu_state *gpu =
    (volatile struct tmu_strided_load_gpu_state *)state->opaque;
uint32_t handle = gpu->handle;

mem_unlock(handle);
mem_free(handle);

state->opaque = 0;
state->prepared = 0;

}

uint32_t tmu_strided_load_runtime_allocations(
const struct tmu_strided_load_state *state)
{
if (!state)
return 0;
return state->runtime_allocations;
}

uint32_t tmu_strided_load_runtime_launches(
const struct tmu_strided_load_state *state)
{
if (!state)
return 0;
return state->runtime_launches;
}

uint32_t tmu_strided_load_runtime_capacity_n(
const struct tmu_strided_load_state *state)
{
if (!state)
return 0;
return state->max_n;
}

uint32_t tmu_strided_load_runtime_capacity_input_words(
const struct tmu_strided_load_state *state)
{
if (!state)
return 0;
return state->max_input_words;
}

