#include "matmul.h"

/* SIMPLE TEST TO COMPARE YOUR KERNEL WITH CPU OUTPUT */
void test_matmul(void)
{
    int i, j;
    volatile struct GPU *gpu;
    gpu_init(&gpu, N);

    uint32_t test_A[N][N];
    uint32_t test_B[N][N];
    uint32_t test_C[N][N];
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            gpu->A[i][j] = i;
            gpu->B[i][j] = i;
            gpu->C[i][j] = 100;
            test_A[i][j] = i;
            test_B[i][j] = i;
            test_C[i][j] = 100;
        }
    }
    printk("STARTING MATMUL\n");
    int start_time = timer_get_usec();
    int iret_matmul = gpu_exec(gpu, NULL);
    int end_time = timer_get_usec();
    int gpu_matmul_time = end_time - start_time;
    printk("MATMUL COMPLETED\n");
    caches_enable();
    int start_time_cpu = timer_get_usec();
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            uint32_t result = 0;
            for (int k = 0; k < N; k++)
            {
                result += test_A[i][k] * test_B[j][k];
            }
            test_C[i][j] = result;
        }
    }
    int end_time_cpu = timer_get_usec();
    int cpu_matmul_time = end_time_cpu - start_time_cpu;
    printk("CPU MATMUL TIME: %d us\n", cpu_matmul_time);
    printk("GPU RESULT:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (gpu->C[i][j] != test_C[i][j]) {
                printk("(%d, %d): GPU: %d, CPU: %d\n", i, j, gpu->C[i][j], test_C[i][j]);
            }
        }
    }
    printk("GPU matmul Time: %d us\n", gpu_matmul_time);
    printk("SPEEDUP: %f\n", (float)cpu_matmul_time / gpu_matmul_time);

    struct profiler_data profiler_data;
    for (i = 0; i < 16; i++) {
        profiler_data.perf_ctr[i].value = 0;
        profiler_data.perf_ctr[i].source_id = 13 + i;
    }
    gpu_exec(gpu, &profiler_data);
    printk("GPU PROFILER DATA:\n");
    profile_dump(&profiler_data, "GPU");
    gpu_release(gpu);
}

void notmain(void)
{
    printk("Testing matmul on GPU...\n");
    
    test_matmul();

    // delay_ms(6000);
}