#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_N 769u
#define MEMORY_PATH_EXPLICIT_TMU_VDW_GUARD 32u
#define MEMORY_PATH_EXPLICIT_TMU_VDW_ACTIVE_QPUS 12u
#define MEMORY_PATH_EXPLICIT_TMU_VDW_LANES 16u
#define MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE (MEMORY_PATH_EXPLICIT_TMU_VDW_ACTIVE_QPUS * MEMORY_PATH_EXPLICIT_TMU_VDW_LANES)
#define MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_WAVES ((MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_N + MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE - 1u) / MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE)
#define MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_COVERAGE_N (MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_WAVES * MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE)
#define MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N (MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_COVERAGE_N + MEMORY_PATH_EXPLICIT_TMU_VDW_GUARD)
#define MEMORY_PATH_EXPLICIT_TMU_VDW_SENTINEL 0xdeadbeefu

static const uint32_t test_sizes[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 768u, 769u
};

static uint32_t a_values[MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N];
static uint32_t b_values[MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N];
static uint32_t out_values[MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N];
static uint32_t expected_values[MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N];

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE - 1u) /
                     MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N; i++) {
        a_values[i] = 0x21000000u + i * 17u + 3u;
        b_values[i] = 0x12000000u + i * 5u + 11u;
        out_values[i] = MEMORY_PATH_EXPLICIT_TMU_VDW_SENTINEL;
        expected_values[i] = i < n ? a_values[i] + b_values[i]
                                   : MEMORY_PATH_EXPLICIT_TMU_VDW_SENTINEL;
    }
}

static int verify_active(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (out_values[i] != expected_values[i]) {
            if (mismatches < 8)
                printk("ERROR: memory_path_explicit_tmu_vdw n=%d i=%d gpu=%x expected=%x\n",
                       (int)n, (int)i, out_values[i], expected_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N; i++) {
        if (out_values[i] != MEMORY_PATH_EXPLICIT_TMU_VDW_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: memory_path_explicit_tmu_vdw sentinel n=%d i=%d gpu=%x expected=%x\n",
                       (int)n, (int)i, out_values[i],
                       MEMORY_PATH_EXPLICIT_TMU_VDW_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N * sizeof(uint32_t);
    vc4_deviceptr_t a_dev = 0;
    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &b_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("memory_path_explicit_tmu_vdw_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 memory_path_explicit_tmu_vdw_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(test_sizes) / sizeof(test_sizes[0]); case_id++) {
        uint32_t n = test_sizes[case_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_inputs(n);
        if (vc4_m2_copy_htod(program, a_dev, a_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            memory_path_explicit_tmu_vdw_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: memory_path_explicit_tmu_vdw_vc4kernel launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(n);
        int sentinels = verify_sentinels(n);
        int checksum = checksum_low16(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("MEMORY_PATH_EXPLICIT_TMU_VDW_CASE case=%d n=%d waves=%d coverage=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, (int)waves,
               (int)(waves * MEMORY_PATH_EXPLICIT_TMU_VDW_ELEMENTS_PER_WAVE),
               mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=memory_path_explicit_tmu_vdw_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(test_sizes) / sizeof(test_sizes[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           MEMORY_PATH_EXPLICIT_TMU_VDW_ACTIVE_QPUS, MEMORY_PATH_EXPLICIT_TMU_VDW_LANES,
           MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_N, MEMORY_PATH_EXPLICIT_TMU_VDW_MAX_COVERAGE_N,
           MEMORY_PATH_EXPLICIT_TMU_VDW_BUFFER_N, checksum_accum, 3,
           (int)(sizeof(test_sizes) / sizeof(test_sizes[0])), elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
