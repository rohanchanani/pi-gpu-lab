#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "stencil2d_5point_naive_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_VPMBASE (V3D_BASE + 0x00504)

#define NUM_UNIFS 9u
#define SCRATCH_SENTINEL (-13579.0f)

struct stencil2d_5point_naive_gpu_state
{
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
uint32_t handle;
uint32_t max_width;
uint32_t max_height;
uint32_t launch_count;
float payload[];
};

static volatile struct stencil2d_5point_naive_gpu_state *g_gpu_state;
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

static size_t scratch_words(uint32_t max_height)
{
return (size_t)max_height * STENCIL2D_5POINT_NAIVE_ROW_STRIDE;
}

static size_t stencil2d_5point_naive_state_size(uint32_t max_height)
{
size_t words = scratch_words(max_height);
return offsetof(struct stencil2d_5point_naive_gpu_state, payload) +
(words + words + STENCIL2D_5POINT_NAIVE_GUARD_FLOATS) * sizeof(float);
}

static float *stencil2d_input_ptr(volatile struct stencil2d_5point_naive_gpu_state *state)
{
return (float *)state->payload;
}

static float *stencil2d_output_ptr(volatile struct stencil2d_5point_naive_gpu_state *state)
{
return (float *)state->payload + scratch_words(state->max_height);
}

int stencil2d_5point_naive_prepare(
struct vc4_runtime *rt,
struct stencil2d_5point_naive_state *state,
uint32_t max_width,
uint32_t max_height)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (max_width > STENCIL2D_5POINT_NAIVE_MAX_WIDTH ||
    max_height > STENCIL2D_5POINT_NAIVE_MAX_HEIGHT)
    return -1;

if (g_gpu_state)
    return -1;

if (vc4_runtime_init(rt) < 0)
    return -1;

uint32_t activeQpus = vc4_runtime_active_qpus(rt);
if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
    return -1;

size_t allocSize = stencil2d_5point_naive_state_size(max_height);
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

volatile struct stencil2d_5point_naive_gpu_state *gpuState =
    (volatile struct stencil2d_5point_naive_gpu_state *)(vc - GPU_BASE);
if (!gpuState)
{
    mem_unlock(handle);
    mem_free(handle);
    return -1;
}

memset((void *)gpuState, 0, allocSize);
gpuState->handle = handle;
gpuState->max_width = max_width;
gpuState->max_height = max_height;
gpuState->launch_count = 0;

memcpy((void *)gpuState->code, shader, sizeof gpuState->code);

for (uint32_t qpu = 0; qpu < VC4_RUNTIME_MAX_QPUS; qpu++)
    gpuState->unif_ptr[qpu] = GPU_BASE + (uint32_t)&gpuState->unif[qpu][0];

PUT32(V3D_VPMBASE, 16u);

g_gpu_state = gpuState;
g_handle = handle;
g_allocations++;

state->rt = rt;
state->prepared = 1;
state->active_qpus = activeQpus;
state->lane_width = vc4_runtime_lane_width();
state->max_width = max_width;
state->max_height = max_height;
state->runtime_allocations = g_allocations;
state->runtime_launches = 0;
state->last_sentinel_mismatches = 0;
state->total_sentinel_mismatches = 0;
return 0;

}

int stencil2d_5point_naive_launch(
struct stencil2d_5point_naive_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height,
float center_weight,
float neighbor_weight)
{
if (!state || !state->prepared || !g_gpu_state)
return -1;
if (width > state->max_width || height > state->max_height)
return -1;

uint32_t total = width * height;
if (total != 0 && (!input || !output))
    return -1;

uint32_t activeQpus = state->active_qpus;
if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
    return -1;

float *gpuInput = stencil2d_input_ptr(g_gpu_state);
float *gpuOutput = stencil2d_output_ptr(g_gpu_state);
size_t inputScratchWords = scratch_words(g_gpu_state->max_height);
size_t outputScratchWords = inputScratchWords;

for (size_t i = 0; i < inputScratchWords; i++)
    gpuInput[i] = 0.0f;

for (size_t i = 0;
     i < outputScratchWords + STENCIL2D_5POINT_NAIVE_GUARD_FLOATS;
     i++)
{
    gpuOutput[i] = SCRATCH_SENTINEL;
}

for (uint32_t y = 0; y < height; y++)
{
    if (width != 0)
    {
        memcpy(gpuInput + (size_t)y * STENCIL2D_5POINT_NAIVE_ROW_STRIDE,
               input + (size_t)y * width,
               (size_t)width * sizeof(float));
    }
}

if (total != 0)
{
    uint32_t inputBus = GPU_BASE + (uint32_t)gpuInput;
    uint32_t outputBus = GPU_BASE + (uint32_t)gpuOutput;

    for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
    {
        g_gpu_state->unif[qpu][0] = inputBus;
        g_gpu_state->unif[qpu][1] = outputBus;
        g_gpu_state->unif[qpu][2] = width;
        g_gpu_state->unif[qpu][3] = height;
        g_gpu_state->unif[qpu][4] = total;
        g_gpu_state->unif[qpu][5] = float_as_u32(center_weight);
        g_gpu_state->unif[qpu][6] = float_as_u32(neighbor_weight);
        g_gpu_state->unif[qpu][7] = qpu;
        g_gpu_state->unif[qpu][8] = activeQpus;
        g_gpu_state->unif_ptr[qpu] =
            GPU_BASE + (uint32_t)&g_gpu_state->unif[qpu][0];
    }

    gpu_fft_base_exec_direct((uint32_t)g_gpu_state->code,
                             (uint32_t *)g_gpu_state->unif_ptr,
                             activeQpus);

    for (uint32_t y = 0; y < height; y++)
    {
        memcpy(output + (size_t)y * width,
               gpuOutput + (size_t)y * STENCIL2D_5POINT_NAIVE_ROW_STRIDE,
               (size_t)width * sizeof(float));
    }
}

uint32_t sentinelMismatches = 0;
for (uint32_t i = 0; i < STENCIL2D_5POINT_NAIVE_GUARD_FLOATS; i++)
{
    if (gpuOutput[outputScratchWords + i] != SCRATCH_SENTINEL)
        sentinelMismatches++;
}

state->last_sentinel_mismatches = sentinelMismatches;
state->total_sentinel_mismatches += sentinelMismatches;

g_gpu_state->launch_count++;
state->runtime_launches = g_gpu_state->launch_count;
return 0;

}

void stencil2d_5point_naive_shutdown(
struct stencil2d_5point_naive_state *state)
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

