#include "rpi.h"
#include "saxpy_launch.h"

#define N 32768
#define SAXPY_EPSILON 0.0001f

static float absf(float value)
{
    return value < 0.0f ? -value : value;
}

static void fill_inputs(float *x, float *y, float *y_initial)
{
    for (int i = 0; i < N; i++)
    {
        x[i] = ((float)(i % 97) * 0.125f) - 3.0f;
        y[i] = ((float)(i % 53) * 0.25f) + 1.0f;
        y_initial[i] = y[i];
    }
}

static void run_cpu_reference(
    float alpha,
    const float *x,
    const float *y_initial,
    float *expected,
    int *cpu_time_usec)
{
    int start_time = timer_get_usec();
    for (int i = 0; i < N; i++)
        expected[i] = alpha * x[i] + y_initial[i];
    int end_time = timer_get_usec();

    *cpu_time_usec = end_time - start_time;
}

static void verify_results(
    const float *y,
    const float *expected,
    int *mismatch_count,
    float *max_abs_diff)
{
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;

    for (int i = 0; i < N; i++)
    {
        float diff = y[i] - expected[i];
        float abs_diff = absf(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > SAXPY_EPSILON)
        {
            if (*mismatch_count < 8)
            {
                printk("ERROR: i=%d gpu=%f cpu=%f diff=%f\n",
                    i, y[i], expected[i], diff);
            }
            (*mismatch_count)++;
        }
    }
}

void notmain(void)
{
    struct vc4_runtime rt;
    static float x[N];
    static float y[N];
    static float y_initial[N];
    static float expected[N];
    const float alpha = 2.5f;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    fill_inputs(x, y, y_initial);

    printk("Running GPU SAXPY...\n");
    int gpu_start = timer_get_usec();
    if (saxpy_launch(&rt, x, y, alpha, N) < 0)
        panic("SAXPY launch failed");
    int gpu_end = timer_get_usec();
    int gpu_time = gpu_end - gpu_start;

    int cpu_time = 0;
    run_cpu_reference(alpha, x, y_initial, expected, &cpu_time);

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(y, expected, &mismatches, &max_abs_diff);

    printk("SAXPY alpha=%f N=%d qpus=%d lanes=%d\n",
        alpha, N, vc4_runtime_active_qpus(&rt), vc4_runtime_lane_width());
    printk("CPU Time: %d us\n", cpu_time);
    printk("GPU Time: %d us\n", gpu_time);
    if (gpu_time > 0)
        printk("Speedup: %f\n", (float)cpu_time / (float)gpu_time);
    else
        printk("Speedup: INF\n");

    printk("Sample results:\n");
    for (int i = 0; i < 4; i++)
    {
        printk("i=%d x=%f y_gpu=%f y_cpu=%f\n",
            i, x[i], y[i], expected[i]);
    }

    if (mismatches)
        panic("SAXPY verification failed: mismatches=%d max_abs_diff=%f",
            mismatches, max_abs_diff);

    printk("SUCCESS: SAXPY verified. max_abs_diff=%f\n", max_abs_diff);
    vc4_runtime_shutdown(&rt);
}
