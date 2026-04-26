#include "rpi.h"
#include "sfu_recip_launch.h"

#define SFU_RECIP_EPSILON 0.0003f
#define CHECKSUM_SCALE 1024.0f
#define SFU_RECIP_MAX_N (VC4_RUNTIME_MAX_QPUS * VC4_RUNTIME_LANE_WIDTH)

static float x_values[SFU_RECIP_MAX_N];
static float out_values[SFU_RECIP_MAX_N];
static float expected_values[SFU_RECIP_MAX_N];

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static float pattern_value(uint32_t index)
{
    switch (index & 7)
    {
    case 0: return 0.5f;
    case 1: return 1.0f;
    case 2: return 2.0f;
    case 3: return 4.0f;
    case 4: return 8.0f;
    case 5: return 16.0f;
    case 6: return 0.25f;
    default: return 32.0f;
    }
}

static void fill_inputs(uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        x_values[i] = pattern_value(i);
        out_values[i] = -99.0f;
        expected_values[i] = 1.0f / x_values[i];
    }
}

static int scaled_checksum(const float *values, uint32_t n)
{
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatches, float *max_abs_diff)
{
    *mismatches = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < n; ++i)
    {
        float diff = out_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > SFU_RECIP_EPSILON)
        {
            if (*mismatches < 8)
            {
                printk("ERROR: i=%u x=%f gpu=%f cpu=%f diff=%f\n",
                       i,
                       x_values[i],
                       out_values[i],
                       expected_values[i],
                       diff);
            }
            (*mismatches)++;
        }
    }
}

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
    uint32_t lane_width = vc4_runtime_lane_width();
    uint32_t n = active_qpus * lane_width;

    if (active_qpus != VC4_RUNTIME_MAX_QPUS)
        panic("Unexpected active QPU count: %u", active_qpus);
    if (n != SFU_RECIP_MAX_N)
        panic("Unexpected test size: %u", n);

    /* This one-vector SFU golden intentionally has no tail path. */
    if (n == 0 || n != active_qpus * lane_width)
        panic("sfu_recip harness chose unsupported n=%u", n);

    fill_inputs(n);

    printk("Running VC4 sfu_recip reference bundle...\n");
    int start = timer_get_usec();
    if (sfu_recip_launch(&rt, x_values, out_values, n) < 0)
        panic("sfu_recip launch failed");
    int end = timer_get_usec();
    int elapsed = end - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);

    int checksum = scaled_checksum(out_values, n);

    printk("sfu_recip qpus=%u lanes=%u n=%u checksum=%d max_abs_diff=%f\n",
           active_qpus,
           lane_width,
           n,
           checksum,
           max_abs_diff);

    for (uint32_t i = 0; i < 4 && i < n; ++i)
    {
        printk("sample i=%u x=%f out=%f expected=%f\n",
               i,
               x_values[i],
               out_values[i],
               expected_values[i]);
    }

    printk("VC4_TEST_RESULT name=sfu_recip status=%s mismatches=%d "
           "active_qpus=%u n=%u checksum=%d max_abs_diff=%f elapsed_usec=%d\n",
           mismatches ? "FAIL" : "PASS",
           mismatches,
           active_qpus,
           n,
           checksum,
           max_abs_diff,
           elapsed);

    if (mismatches)
        panic("sfu_recip verification failed: mismatches=%d max_abs_diff=%f",
              mismatches,
              max_abs_diff);

    vc4_runtime_shutdown(&rt);
}
