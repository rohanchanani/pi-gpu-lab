#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "conv1d_3tap_launch.h"
#include "shader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define V3D_BASE 0x20C00000
#define V3D_VPMBASE (V3D_BASE + 0x00504)

#define NUM_UNIFS 8u

struct conv1d_3tap_gpu_state
{
uint32_t code[sizeof(shader) / sizeof(uint32_t)];
uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
uint32_t handle;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t launch_count;
float payload[];
};

static volatile struct conv1d_3tap_gpu_state *g_gpu_state;
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

static size_t conv1d_3tap_state_size(uint32_t padded_capacity_n)
{
return offsetof(struct conv1d_3tap_gpu_state, payload) +
(size_t)(padded_capacity_n + padded_capacity_n + CONV1D_3TAP_GUARD_FLOATS) *
sizeof(float);
}

static float *conv1d_3tap_input_ptr(volatile struct conv1d_3tap_gpu_state *state)
{
return (float *)state->payload;
}

static float *conv1d_3tap_output_ptr(volatile struct conv1d_3tap_gpu_state *state)
{
return (float *)state->payload + state->padded_capacity_n;
}

int conv1d_3tap_prepare(
struct vc4_runtime *rt,
struct conv1d_3tap_state *state,
uint32_t max_n)
{
if (!rt || !state)
return -1;

memset(state, 0, sizeof *state);

if (max_n > CONV1D_3TAP_MAX_N)
    return -1;

if (g_gpu_state)
    return -1;

if (vc4_runtime_init(rt) < 0)
    return -1;

uint32_t activeQpus = vc4_runtime_active_qpus(rt);
if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
    return -1;

uint32_t paddedCapacity = round_up_to_lane_width(max_n);
if (paddedCapacity < max_n)
    return -1;

size_t allocSize = conv1d_3tap_state_size(paddedCapacity);
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

volatile struct conv1d_3tap_gpu_state *gpuState =
    (volatile struct conv1d_3tap_gpu_state *)(vc - GPU_BASE);
if (!gpuState)
{
    mem_unlock(handle);
    mem_free(handle);
    return -1;
}

memset((void *)gpuState, 0, allocSize);
gpuState->handle = handle;
gpuState->max_n = max_n;
gpuState->padded_capacity_n = paddedCapacity;
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
state->max_n = max_n;
state->padded_capacity_n = paddedCapacity;
state->runtime_allocations = g_allocations;
state->runtime_launches = 0;
state->last_sentinel_mismatches = 0;
state->total_sentinel_mismatches = 0;
return 0;

}

int conv1d_3tap_launch(
struct conv1d_3tap_state *state,
const float *x,
float *out,
uint32_t n,
float c0,
float c1,
float c2)
{
if (!state || !state->prepared || !g_gpu_state)
return -1;
if (n > state->max_n)
return -1;
if (n != 0 && (!x || !out))
return -1;

uint32_t activeQpus = state->active_qpus;
if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
    return -1;

uint32_t paddedN = round_up_to_lane_width(n);
if (paddedN < n || paddedN > g_gpu_state->padded_capacity_n)
    return -1;

float *gpuInput = conv1d_3tap_input_ptr(g_gpu_state);
float *gpuOutput = conv1d_3tap_output_ptr(g_gpu_state);

if (n != 0)
    memcpy(gpuInput, x, n * sizeof(float));

for (uint32_t i = n; i < paddedN; i++)
    gpuInput[i] = 0.0f;

for (uint32_t i = 0;
     i < g_gpu_state->padded_capacity_n + CONV1D_3TAP_GUARD_FLOATS;
     i++)
{
    gpuOutput[i] = -24680.0f;
}

uint32_t inputBus = GPU_BASE + (uint32_t)gpuInput;
uint32_t outputBus = GPU_BASE + (uint32_t)gpuOutput;

for (uint32_t qpu = 0; qpu < activeQpus; qpu++)
{
    g_gpu_state->unif[qpu][0] = inputBus;
    g_gpu_state->unif[qpu][1] = outputBus;
    g_gpu_state->unif[qpu][2] = n;
    g_gpu_state->unif[qpu][3] = float_as_u32(c0);
    g_gpu_state->unif[qpu][4] = float_as_u32(c1);
    g_gpu_state->unif[qpu][5] = float_as_u32(c2);
    g_gpu_state->unif[qpu][6] = qpu;
    g_gpu_state->unif[qpu][7] = activeQpus;
    g_gpu_state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&g_gpu_state->unif[qpu][0];
}

gpu_fft_base_exec_direct((uint32_t)g_gpu_state->code,
                         (uint32_t *)g_gpu_state->unif_ptr,
                         activeQpus);

uint32_t sentinelMismatches = 0;
for (uint32_t i = n;
     i < n + CONV1D_3TAP_GUARD_FLOATS &&
     i < g_gpu_state->padded_capacity_n + CONV1D_3TAP_GUARD_FLOATS;
     i++)
{
    if (gpuOutput[i] != -24680.0f)
        sentinelMismatches++;
}

state->last_sentinel_mismatches = sentinelMismatches;
state->total_sentinel_mismatches += sentinelMismatches;

if (n != 0)
    memcpy(out, gpuOutput, n * sizeof(float));

g_gpu_state->launch_count++;
state->runtime_launches = g_gpu_state->launch_count;
return 0;

}

void conv1d_3tap_shutdown(struct conv1d_3tap_state *state)
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

