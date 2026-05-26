#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define WARP_PREFIX_SUM_LANE_WIDTH 16u
#define WARP_PREFIX_SUM_MAX_QPUS 12u
#define WARP_PREFIX_SUM_GUARD_WORDS 32u
#define WARP_PREFIX_SUM_SENTINEL 0xdeadbeefu
#define WARP_PREFIX_SUM_MAX_N 511u
#define WARP_PREFIX_SUM_BUFFER_N (WARP_PREFIX_SUM_MAX_N + WARP_PREFIX_SUM_GUARD_WORDS)

enum {
    WPS_CASES = 14
};

static const uint32_t test_sizes[WPS_CASES] = {
    0u, 1u, 2u, 3u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 511u,
};

static uint32_t input_values[WARP_PREFIX_SUM_BUFFER_N];
static uint32_t out_values[WARP_PREFIX_SUM_BUFFER_N];
static uint32_t expected_values[WARP_PREFIX_SUM_BUFFER_N];

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    if (a == 0u)
        return 0u;
    return (a + b - 1u) / b;
}

static uint32_t make_input_value(uint32_t i) {
    return ((i * 7u + 3u) % 11u) + 1u;
}

static void fill_buffers(uint32_t n) {
    for (uint32_t i = 0; i < WARP_PREFIX_SUM_BUFFER_N; i++) {
        input_values[i] = 0u;
        out_values[i] = WARP_PREFIX_SUM_SENTINEL;
        expected_values[i] = WARP_PREFIX_SUM_SENTINEL;
    }

    for (uint32_t i = 0; i < n; i++)
        input_values[i] = make_input_value(i);
}

static void run_cpu_reference(uint32_t n) {
    uint32_t vectors = ceil_div_u32(n, WARP_PREFIX_SUM_LANE_WIDTH);

    for (uint32_t v = 0; v < vectors; v++) {
        uint32_t base = v * WARP_PREFIX_SUM_LANE_WIDTH;
        uint32_t remaining = n - base;
        uint32_t active = remaining < WARP_PREFIX_SUM_LANE_WIDTH ? remaining : WARP_PREFIX_SUM_LANE_WIDTH;
        uint32_t sum = 0u;

        for (uint32_t lane = 0; lane < active; lane++) {
            sum += input_values[base + lane];
            expected_values[base + lane] = sum;
        }
    }
}

static void verify_results(uint32_t n, int *mismatches) {
    *mismatches = 0;

    for (uint32_t i = 0; i < n; i++) {
        if (out_values[i] != expected_values[i]) {
            if (*mismatches < 8) {
                uint32_t vector = i / WARP_PREFIX_SUM_LANE_WIDTH;
                uint32_t lane = i % WARP_PREFIX_SUM_LANE_WIDTH;
                printk("ERROR: vector=%d lane=%d i=%d gpu=%x cpu=%x\n",
                       (int)vector, (int)lane, (int)i, out_values[i], expected_values[i]);
            }
            (*mismatches)++;
        }
    }
}

static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;

    for (uint32_t i = n; i < n + WARP_PREFIX_SUM_GUARD_WORDS && i < WARP_PREFIX_SUM_BUFFER_N; i++) {
        if (out_values[i] != WARP_PREFIX_SUM_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: sentinel changed i=%d value=%x expected=%x\n",
                       (int)i, out_values[i], WARP_PREFIX_SUM_SENTINEL);
            mismatches++;
        }
    }

    return mismatches;
}

static uint32_t checksum_u32(const uint32_t *values, uint32_t n) {
    uint32_t checksum = 0u;
    for (uint32_t i = 0; i < n; i++)
        checksum += values[i];
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = WARP_PREFIX_SUM_BUFFER_N * sizeof(uint32_t);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 || vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("warp_prefix_sum device allocation failed");

    uint32_t active_qpus = WARP_PREFIX_SUM_MAX_QPUS;
    uint32_t lane_width = WARP_PREFIX_SUM_LANE_WIDTH;
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1u, 1u);

    printk("WARP_PREFIX_SUM_RUNTIME_SETUP max_n=%d allocations=%d capacity=%d code_uploads=%d\n",
           WARP_PREFIX_SUM_MAX_N,
           (int)warp_prefix_sum_runtime_allocations(),
           (int)warp_prefix_sum_runtime_capacity(),
           (int)warp_prefix_sum_runtime_code_uploads());

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checked_elements = 0u;
    uint32_t checksum_accum = 0u;
    uint32_t expected_checksum_accum = 0u;
    int start = timer_get_usec();

    for (uint32_t case_index = 0; case_index < WPS_CASES; case_index++) {
        uint32_t n = test_sizes[case_index];
        uint32_t vectors = ceil_div_u32(n, WARP_PREFIX_SUM_LANE_WIDTH);

        fill_buffers(n);
        run_cpu_reference(n);

        if (vc4_m2_copy_htod(program, input_dev, input_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            warp_prefix_sum_launch(program, grid, block, input_dev, out_dev, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: warp_prefix_sum launch/copy failed case=%d n=%d\n",
                   (int)case_index, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        verify_results(n, &mismatches);
        int case_sentinel_mismatches = verify_sentinel_tail(n);
        uint32_t checksum = checksum_u32(out_values, n);
        uint32_t expected_checksum = checksum_u32(expected_values, n);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d gpu=%x cpu=%x\n",
                   (int)case_index, checksum, expected_checksum);
            mismatches++;
        }

        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel_mismatches;
        checked_elements += n;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;

        printk("WARP_PREFIX_SUM_CASE case=%d n=%d vectors=%d mismatches=%d sentinel_mismatches=%d checksum=%d expected_checksum=%d launches=%d allocations=%d\n",
               (int)case_index, (int)n, (int)vectors, mismatches,
               case_sentinel_mismatches, (int)checksum, (int)expected_checksum,
               (int)warp_prefix_sum_runtime_launches(), (int)warp_prefix_sum_runtime_allocations());
    }

    int elapsed = timer_get_usec() - start;
    uint32_t runtime_allocations = warp_prefix_sum_runtime_allocations();
    uint32_t runtime_launches = warp_prefix_sum_runtime_launches();
    uint32_t runtime_capacity = warp_prefix_sum_runtime_capacity();
    uint32_t code_uploads = warp_prefix_sum_runtime_code_uploads();
    uint32_t recorded_launch_failures = warp_prefix_sum_runtime_launch_failures();
    const char *status =
        (total_mismatches == 0 &&
         sentinel_mismatches == 0 &&
         launch_failures == 0 &&
         recorded_launch_failures == 0u &&
         checked_elements == 1237u &&
         checksum_accum == expected_checksum_accum &&
         runtime_allocations == 1u &&
         runtime_launches == WPS_CASES &&
         runtime_capacity == 65536u &&
         code_uploads == 1u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=warp_prefix_sum status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d expected_checksum_accum=%d runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status, WPS_CASES, (int)checked_elements, total_mismatches, sentinel_mismatches,
           launch_failures, (int)recorded_launch_failures, (int)active_qpus, (int)lane_width,
           WARP_PREFIX_SUM_MAX_N, (int)checksum_accum, (int)expected_checksum_accum,
           (int)runtime_allocations, (int)runtime_launches, (int)runtime_capacity, (int)code_uploads, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
