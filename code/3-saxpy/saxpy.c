#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "saxpy.h"
#include "mailbox.h"

static void saxpy_gpu_prepare(volatile struct saxpy_gpu **gpu)
{
    uint32_t handle, vc;
    volatile struct saxpy_gpu *ptr;

    if (qpu_enable(1))
        panic("Failed to enable GPU");

    handle = mem_alloc(sizeof(struct saxpy_gpu), 4096, GPU_MEM_FLG);
    if (!handle)
    {
        qpu_enable(0);
        panic("Failed to allocate GPU memory");
    }

    vc = mem_lock(handle);

    ptr = (volatile struct saxpy_gpu *)(vc - GPU_BASE);
    if (ptr == NULL)
    {
        mem_free(handle);
        mem_unlock(handle);
        qpu_enable(0);
        panic("Failed to convert handle to CPU address");
    }

    ptr->handle = handle;
    ptr->mail[0] = GPU_BASE + (uint32_t)&ptr->code;
    ptr->mail[1] = GPU_BASE + (uint32_t)&ptr->unif;

    *gpu = ptr;
}

static uint32_t saxpy_gpu_execute(volatile struct saxpy_gpu *gpu)
{
    return gpu_fft_base_exec_direct(
        (uint32_t)gpu->mail[0],
        (uint32_t *)gpu->unif_ptr,
        NUM_QPUS);
}

uint32_t float_as_u32(float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

void saxpy_init(volatile struct saxpy_gpu **gpu, float alpha)
{
    if (N % (NUM_QPUS * LANES_PER_QPU) != 0)
        panic("N must be divisible by NUM_QPUS * LANES_PER_QPU");

    saxpy_gpu_prepare(gpu);

    volatile struct saxpy_gpu *ptr = *gpu;
    memcpy((void *)ptr->code, saxpyshader, sizeof ptr->code);

    for (int i = 0; i < NUM_QPUS; i++)
    {
        ptr->unif[i][0] = GPU_BASE + (uint32_t)&ptr->x[0];
        ptr->unif[i][1] = GPU_BASE + (uint32_t)&ptr->y[0];
        ptr->unif[i][2] = float_as_u32(alpha);
        ptr->unif[i][3] = NUM_QPUS;
        ptr->unif[i][4] = i;
        ptr->unif[i][5] = NUM_ITERS;
        ptr->unif_ptr[i] = GPU_BASE + (uint32_t)&ptr->unif[i];
    }
}

int saxpy_exec(volatile struct saxpy_gpu *gpu)
{
    int start_time = timer_get_usec();
    saxpy_gpu_execute(gpu);
    int end_time = timer_get_usec();

    return end_time - start_time;
}

void saxpy_release(volatile struct saxpy_gpu *gpu)
{
    uint32_t handle = gpu->handle;
    mem_unlock(handle);
    mem_free(handle);
    qpu_enable(0);
}
