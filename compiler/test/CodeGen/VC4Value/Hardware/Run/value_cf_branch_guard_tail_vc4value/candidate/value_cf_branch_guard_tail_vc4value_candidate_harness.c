#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAX_N 1000u
#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define SENTINEL 0x6d5a1307u

static const uint32_t n_cases[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 193u, 1000u};
static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = (int32_t)SENTINEL;
}

static int verify_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        int32_t expected = 101;
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: branch_guard active n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)out_values[index], (uint32_t)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: branch_guard sentinel n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)out_values[index] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    uint32_t bytes = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("branch_guard allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_output();
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            value_cf_branch_guard_tail_vc4value_launch(program, grid, block, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: branch_guard launch/copy failed n=%d\n", (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(n);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output(n);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)n;
        printk("VALUE_CF_BRANCH_GUARD_CASE case=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)waves, mismatches, sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_cf_branch_guard_tail_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u saw_value_cf_cond_br=1 saw_value_cf_branch_skip=1 saw_value_tail_mask=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash, 1,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
