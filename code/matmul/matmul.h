// TODO: SWAP THESE

#include "matmulshader.h"

#include "rpi.h"
#include <stdint.h>
#include "mailbox.h"

#define GPU_MEM_FLG 0xC // cached=0xC; direct=0x4
#define GPU_BASE 0x40000000

// FEEL FREE TO CHANGE THIS VALUE AND SEE HOW SPEEDUP CHANGES - WILL NEED TO BE
// DIVISIBLE BY 16 (good extension is write a kernel that doesn't need this)
#define N 64
// TODO: AFTER YOU DECIDE WHAT YOUR UNIFORMS SHOULD BE, SET THIS CONSTANT
// (Ours is 4, yours doesn't have to be)
// ALSO MAKE SURE YOU SWAP SHADER HEADER FILES (ABOVE)
#define NUM_UNIFS 6
#define NUM_QPUS 4

struct GPU
{
	uint32_t A[N][N];
	uint32_t B[N][N];
	uint32_t C[N][N];
	uint32_t code[sizeof(matmulshader) / sizeof(uint32_t)];
	uint32_t unif[NUM_QPUS][NUM_UNIFS];
	uint32_t unif_ptr[NUM_QPUS];
	uint32_t mail[2];
	uint32_t handle;
};

void gpu_init(volatile struct GPU **gpu, int n);

int gpu_exec(volatile struct GPU *gpu, struct profiler_data *profiler_data);

void gpu_release(volatile struct GPU *gpu);