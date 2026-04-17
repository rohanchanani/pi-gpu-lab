#include "kernel_launch.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000
#define NUM_UNIFS 6

extern const uint32_t saxpyshader[];
extern const size_t saxpyshader_word_count;

uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags);
uint32_t mem_free(uint32_t handle);
uint32_t mem_lock(uint32_t handle);
uint32_t mem_unlock(uint32_t handle);
unsigned gpu_fft_base_exec_direct(uint32_t code, uint32_t unifs[], int num_qpus);
uint32_t vc4_runtime_lane_width(void);
uint32_t vc4_runtime_active_qpus(const struct vc4_runtime *rt);

enum {
  VC4_RUNTIME_MAX_QPUS = 12,
};

struct saxpy_launch_state {
  uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
  uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
  uint32_t handle;
  uint32_t payload[];
};

static uint32_t vc4_pack_f32(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint32_t *saxpy_code_ptr(volatile struct saxpy_launch_state *state) {
  return (uint32_t *)state->payload;
}

static size_t saxpy_code_bytes(void) {
  return saxpyshader_word_count * sizeof(uint32_t);
}

static size_t saxpy_payload_offset_bytes(void) {
  return saxpy_code_bytes();
}

static float *saxpy_arg0_ptr(volatile struct saxpy_launch_state *state, uint32_t extent) {
  return (float *)((uint8_t *)state->payload + saxpy_payload_offset_bytes());
}

static float *saxpy_arg1_ptr(volatile struct saxpy_launch_state *state, uint32_t extent) {
  return (float *)((uint8_t *)state->payload + saxpy_payload_offset_bytes() + (size_t)extent * sizeof(float));
}

static size_t saxpy_launch_state_size(uint32_t extent) {
  return offsetof(struct saxpy_launch_state, payload) + saxpy_payload_offset_bytes() + (size_t)extent * sizeof(float) + (size_t)extent * sizeof(float);
}

int saxpy_launch(struct vc4_runtime *rt, float * arg0, float * arg1, float arg2, uint32_t arg3) {
  uint32_t activeQpus = vc4_runtime_active_qpus(rt);
  uint32_t laneWidth = vc4_runtime_lane_width();

  if (!rt)
    return -1;
  if (!arg0)
    return -1;
  if (!arg1)
    return -1;
  if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)
    return -1;
  (void)laneWidth;

  size_t allocSize = saxpy_launch_state_size(arg3);
  uint32_t handle = mem_alloc((uint32_t)allocSize, 4096, GPU_MEM_FLG);
  if (!handle)
    return -1;

  uint32_t vc = mem_lock(handle);
  if (!vc) {
    mem_free(handle);
    return -1;
  }

  volatile struct saxpy_launch_state *state =
      (volatile struct saxpy_launch_state *)(vc - GPU_BASE);
  if (!state) {
    mem_unlock(handle);
    mem_free(handle);
    return -1;
  }

  state->handle = handle;
  memcpy((void *)saxpy_code_ptr(state), saxpyshader, saxpy_code_bytes());

  float *gpuArg0 = saxpy_arg0_ptr(state, arg3);
  memcpy(gpuArg0, arg0, (size_t)arg3 * sizeof(float));
  uint32_t gpuArg0Addr = GPU_BASE + (uint32_t)gpuArg0;

  float *gpuArg1 = saxpy_arg1_ptr(state, arg3);
  memcpy(gpuArg1, arg1, (size_t)arg3 * sizeof(float));
  uint32_t gpuArg1Addr = GPU_BASE + (uint32_t)gpuArg1;

  for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {
    /* user uniforms first, builtin suffix second */
    state->unif[qpu][0] = gpuArg0Addr;
    state->unif[qpu][1] = gpuArg1Addr;
    state->unif[qpu][2] = vc4_pack_f32(arg2);
    state->unif[qpu][3] = arg3;
    state->unif[qpu][4] = qpu;
    state->unif[qpu][5] = activeQpus;
    state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];
  }

  gpu_fft_base_exec_direct((uint32_t)saxpy_code_ptr(state), (uint32_t *)state->unif_ptr, activeQpus);
  memcpy(arg1, gpuArg1, (size_t)arg3 * sizeof(float));

  mem_unlock(handle);
  mem_free(handle);
  return 0;
}
