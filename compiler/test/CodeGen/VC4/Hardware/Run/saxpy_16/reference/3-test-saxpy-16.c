#include "rpi.h"
#include "saxpy_16_launch.h"

#define SAXPY_16_CHUNKS_PER_QPU 4
#define SAXPY_16_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define EXPECTED_CHECKSUM 11592704

#define SAXPY_16_MAX_N (VC4_RUNTIME_MAX_QPUS * VC4_RUNTIME_LANE_WIDTH * SAXPY_16_CHUNKS_PER_QPU)

static float x_values[SAXPY_16_MAX_N];
static float y_values[SAXPY_16_MAX_N];
static float y_initial[SAXPY_16_MAX_N];
static float expected_values[SAXPY_16_MAX_N];

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static void fill_inputs(uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
    {
        x_values[i] = ((float)(i % 97) * 0.125f) - 3.0f;
        y_values[i] = ((float)(i % 53) * 0.25f) + 1.0f;
        y_initial[i] = y_values[i];
    }
}

static void run_cpu_reference(float alpha, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        expected_values[i] = alpha * x_values[i] + y_initial[i];
}

static int scaled_checksum(const float *values, uint32_t n)
{
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff)
{
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < n; i++)
    {
        float diff = y_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > SAXPY_16_EPSILON)
        {
            if (*mismatch_count < 8)
            {
                printk("ERROR: i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)i,
                       y_values[i],
                       expected_values[i],
                       diff);
            }
            (*mismatch_count)++;
        }
    }
}

void notmain(void)
{
    struct vc4_runtime rt;
    const float alpha = 2.5f;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t activeQpus = vc4_runtime_active_qpus(&rt);
    uint32_t laneWidth = vc4_runtime_lane_width();
    uint32_t n = activeQpus * laneWidth * SAXPY_16_CHUNKS_PER_QPU;

    if (activeQpus != VC4_RUNTIME_MAX_QPUS)
        panic("Unexpected active QPU count: %d", (int)activeQpus);
    if (n != SAXPY_16_MAX_N)
        panic("Unexpected test size: %d", (int)n);

    /*
     * This check belongs in the handwritten/upstream-style harness, not in the
     * generated-style launcher. The saxpy_16 kernel has no tail mask, so this
     * golden intentionally runs only an exact multiple of one round of all
     * active QPUs.
     */
    if (n == 0 || (n % (activeQpus * laneWidth)) != 0)
        panic("saxpy_16 harness chose unsupported n=%d", (int)n);

    fill_inputs(n);
    run_cpu_reference(alpha, n);

    printk("Running VC4 saxpy_16 reference bundle...\n");
    int start = timer_get_usec();
    if (saxpy_16_launch(&rt, x_values, y_values, alpha, n) < 0)
        panic("saxpy_16 launch failed");
    int end = timer_get_usec();
    int elapsed = end - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);

    int checksum = scaled_checksum(y_values, n);
    int expected_checksum = scaled_checksum(expected_values, n);
    if (checksum != expected_checksum)
        panic("checksum mismatch: gpu=%d cpu=%d", checksum, expected_checksum);
    if (checksum != EXPECTED_CHECKSUM)
        panic("unexpected checksum: got=%d expected=%d", checksum, EXPECTED_CHECKSUM);

    printk("saxpy_16 alpha=%f qpus=%d lanes=%d n=%d chunks_per_qpu=%d checksum=%d\n",
           alpha,
           (int)activeQpus,
           (int)laneWidth,
           (int)n,
           SAXPY_16_CHUNKS_PER_QPU,
           checksum);
    for (int i = 0; i < 4; i++)
    {
        printk("sample i=%d x=%f y_gpu=%f y_cpu=%f\n",
               i,
               x_values[i],
               y_values[i],
               expected_values[i]);
    }

    if (mismatches)
        panic("saxpy_16 verification failed: mismatches=%d max_abs_diff=%f",
              mismatches,
              max_abs_diff);

    printk("VC4_TEST_RESULT name=saxpy_16 status=PASS mismatches=%d active_qpus=%d n=%d chunks_per_qpu=%d checksum=%d max_abs_diff=%f elapsed_usec=%d\n",
           mismatches,
           (int)activeQpus,
           (int)n,
           SAXPY_16_CHUNKS_PER_QPU,
           checksum,
           max_abs_diff,
           elapsed);

    vc4_runtime_shutdown(&rt);
}
