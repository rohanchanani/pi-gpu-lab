#ifndef SAXPY_H
#define SAXPY_H

#include "rpi.h"
#include <stdint.h>
#include "mailbox.h"
#include "saxpyshader.h"

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define NUM_QPUS 8
#define LANES_PER_QPU 16
#define N 32768
#define NUM_ITERS (N / (NUM_QPUS * LANES_PER_QPU))
#define NUM_UNIFS 6
#define SAXPY_EPSILON 0.0001f

struct saxpy_gpu
{
    float x[N];
    float y[N];
    uint32_t code[sizeof(saxpyshader) / sizeof(uint32_t)];
    uint32_t unif[NUM_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[NUM_QPUS];
    uint32_t mail[2];
    uint32_t handle;
};

void saxpy_init(volatile struct saxpy_gpu **gpu, float alpha);
int saxpy_exec(volatile struct saxpy_gpu *gpu);
void saxpy_release(volatile struct saxpy_gpu *gpu);
uint32_t float_as_u32(float value);

#endif
