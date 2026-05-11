#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VECTOR_STORE_SMOKE_MAX_N 192u
#define VECTOR_STORE_SMOKE_GUARD 32u
#define VECTOR_STORE_SMOKE_BUFFER_N (VECTOR_STORE_SMOKE_MAX_N + VECTOR_STORE_SMOKE_GUARD)
#define VECTOR_STORE_SMOKE_SENTINEL 0xdeadbeefu
#define VECTOR_STORE_SMOKE_ACTIVE_QPUS 12u
#define VECTOR_STORE_SMOKE_LANE_WIDTH 16u

static uint32_t out_values[VECTOR_STORE_SMOKE_BUFFER_N];
static const uint32_t test_sizes[] = {0u, 1u, 16u, 17u, 192u};

static uint32_t expected_value(uint32_t case_id, uint32_t i) {
    return 0x51000000u | ((case_id & 0xffu) << 16) | (i & 0xffffu);
}

static void fill_host_buffer(void) {
    for (uint32_t i = 0; i < VECTOR_STORE_SMOKE_BUFFER_N; i++)
        out_values[i] = VECTOR_STORE_SMOKE_SENTINEL;
}

static int verify_results(uint32_t case_id, uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t expected = expected_value(case_id, i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: vector_store case=%d i=%d gpu=%x expected=%x\n", (int)case_id, (int)i, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + VECTOR_STORE_SMOKE_GUARD && i < VECTOR_STORE_SMOKE_BUFFER_N; i++) {
        if (out_values[i] != VECTOR_STORE_SMOKE_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: vector_store sentinel changed i=%d value=%x expected=%x\n", (int)i, out_values[i], VECTOR_STORE_SMOKE_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(const uint32_t *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    const uint32_t activeQpus = VECTOR_STORE_SMOKE_ACTIVE_QPUS;
    const uint32_t laneWidth = VECTOR_STORE_SMOKE_LANE_WIDTH;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = VECTOR_STORE_SMOKE_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vector_store_smoke device allocation failed");

    const uint32_t case_count = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 vector_store_smoke M2 candidate bundle...\n");
    printk("VECTOR_STORE_SMOKE_SETUP max_n=%d active_qpus=%d lanes=%d allocations=%d\n",
           (int)VECTOR_STORE_SMOKE_MAX_N, (int)activeQpus, (int)laneWidth, 1);

    for (uint32_t case_index = 0; case_index < case_count; case_index++) {
        uint32_t n = test_sizes[case_index];
        uint32_t case_id = case_index;
        fill_host_buffer();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            vector_store_smoke_launch(program, grid, block, out_dev, n, case_id) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: vector_store_smoke launch/copy failed case=%d n=%d\n", (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(case_id, n);
        int case_sentinel_mismatches = verify_sentinel_tail(n);
        int checksum = checksum_low16(out_values, n);
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel_mismatches;
        checksum_accum += checksum;
        printk("VECTOR_STORE_SMOKE_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, mismatches, case_sentinel_mismatches, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vector_store_smoke status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)case_count, total_mismatches, sentinel_mismatches, launch_failures, (int)activeQpus, (int)laneWidth, VECTOR_STORE_SMOKE_MAX_N, checksum_accum, 1, (int)case_count, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
