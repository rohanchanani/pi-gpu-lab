#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define CONV1D_3TAP_CASES 13u
#define CONV1D_3TAP_MAX_N 511u
#define CONV1D_3TAP_GUARD_FLOATS 32u
#define CONV1D_3TAP_BUFFER_N (CONV1D_3TAP_MAX_N + CONV1D_3TAP_GUARD_FLOATS)
#define CONV1D_3TAP_SENTINEL (-24680.0f)
#define CHECKSUM_SCALE 4096.0f
#define CONV1D_3TAP_ACTIVE_QPUS 12u
#define CONV1D_3TAP_LANES 16u

static const uint32_t test_sizes[CONV1D_3TAP_CASES] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 511u,
};

static float x_values[CONV1D_3TAP_BUFFER_N];
static float out_values[CONV1D_3TAP_BUFFER_N];
static float expected_values[CONV1D_3TAP_BUFFER_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float input_value(uint32_t i) {
    return ((float)((i * 13u + 7u) % 97u) * 0.0625f) - 3.0f;
}

static void fill_case(void) {
    for (uint32_t i = 0; i < CONV1D_3TAP_BUFFER_N; i++) {
        x_values[i] = input_value(i);
        out_values[i] = CONV1D_3TAP_SENTINEL;
        expected_values[i] = CONV1D_3TAP_SENTINEL;
    }
}

static void run_cpu_reference(uint32_t n, float c0, float c1, float c2) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t left = (i == 0u) ? 0u : i - 1u;
        uint32_t right = (i + 1u >= n) ? n - 1u : i + 1u;
        expected_values[i] =
            (c0 * x_values[left] + c1 * x_values[i]) + c2 * x_values[right];
    }
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    const float c0 = -0.25f;
    const float c1 = 1.5f;
    const float c2 = 0.75f;
    struct vc4_program *program = 0;
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    const uint32_t bytes = CONV1D_3TAP_BUFFER_N * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t checked_elements = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("conv1d_3tap device allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(CONV1D_3TAP_ACTIVE_QPUS * CONV1D_3TAP_LANES, 1u, 1u);

    printk("CONV1D_3TAP_RUNTIME_SETUP max_n=%d allocations=1\n",
           CONV1D_3TAP_MAX_N);

    for (uint32_t case_id = 0; case_id < CONV1D_3TAP_CASES; case_id++) {
        uint32_t n = test_sizes[case_id];
        uint32_t mismatches = 0, case_sentinel = 0;
        float max_abs_diff = 0.0f;
        int checksum, expected_checksum;

        fill_case();
        run_cpu_reference(n, c0, c1, c2);

        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            conv1d_3tap_launch(program, grid, block, x_dev, out_dev, n, c0, c1, c2) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            launch_failures++;
            printk("CONV1D_3TAP_CASE case=%d n=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)n, (int)(case_id + 1u));
            continue;
        }

        for (uint32_t i = 0; i < n; i++) {
            float diff = out_values[i] - expected_values[i];
            float adiff = absf_local(diff);
            if (adiff > max_abs_diff)
                max_abs_diff = adiff;
            checked_elements++;
            if (out_values[i] != expected_values[i]) {
                if (mismatches < 8u)
                    printk("ERROR: case=%d n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                           (int)case_id, (int)n, (int)i, out_values[i],
                           expected_values[i], diff);
                mismatches++;
            }
        }

        for (uint32_t i = n; i < n + CONV1D_3TAP_GUARD_FLOATS && i < CONV1D_3TAP_BUFFER_N; i++) {
            if (out_values[i] != CONV1D_3TAP_SENTINEL) {
                if (case_sentinel < 8u)
                    printk("ERROR: sentinel case=%d n=%d i=%d value=%f expected=%f\n",
                           (int)case_id, (int)n, (int)i, out_values[i],
                           CONV1D_3TAP_SENTINEL);
                case_sentinel++;
            }
        }

        checksum = scaled_checksum(out_values, n);
        expected_checksum = scaled_checksum(expected_values, n);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d n=%d gpu=%d cpu=%d\n",
                   (int)case_id, (int)n, checksum, expected_checksum);
            mismatches++;
        }

        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;

        printk("CONV1D_3TAP_CASE case=%d n=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)n, (int)n, (int)mismatches, (int)case_sentinel,
               checksum, max_abs_diff, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u &&
                          launch_failures == 0u && checked_elements == 1234u &&
                          global_max_abs_diff == 0.0f)
                             ? "PASS"
                             : "FAIL";

    printk("VC4_TEST_RESULT name=conv1d_3tap status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d checksum_accum=%d elapsed_usec=%d\n",
           status, CONV1D_3TAP_CASES, (int)checked_elements, (int)total_mismatches,
           (int)sentinel_mismatches, (int)launch_failures, CONV1D_3TAP_ACTIVE_QPUS,
           CONV1D_3TAP_LANES, CONV1D_3TAP_MAX_N, global_max_abs_diff, 1,
           CONV1D_3TAP_CASES, checksum_accum, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
