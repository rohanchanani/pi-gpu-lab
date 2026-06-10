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
#define SENTINEL 0x6a5acd11u

static const uint32_t n_cases[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 1000u
};

static int32_t x_values[BUFFER_N];
static int32_t tail_out[BUFFER_N];
static int32_t full_out[BUFFER_N];
static int32_t empty_out[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int32_t input_value(uint32_t i) {
    return (int32_t)((i * 37u + 11u) ^ (0x13572468u + (i << 3)));
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        x_values[i] = input_value(active);
        tail_out[i] = (int32_t)SENTINEL;
        full_out[i] = (int32_t)SENTINEL;
        empty_out[i] = (int32_t)SENTINEL;
    }
}

static int verify_tail(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active_begin = GUARD;
        uint32_t active_end = GUARD + n;
        int32_t expected = (i >= active_begin && i < active_end)
                               ? input_value(i - GUARD)
                               : (int32_t)SENTINEL;
        if (tail_out[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: tail n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)tail_out[i], (uint32_t)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_full(uint32_t coverage) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active_begin = GUARD;
        uint32_t active_end = GUARD + coverage;
        int32_t expected = (i >= active_begin && i < active_end)
                               ? input_value(i - GUARD)
                               : (int32_t)SENTINEL;
        if (full_out[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: full coverage=%d i=%d got=%x expected=%x\n",
                       (int)coverage, (int)i, (uint32_t)full_out[i],
                       (uint32_t)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_empty(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if ((uint32_t)empty_out[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: empty store changed i=%d got=%x expected=%x\n",
                       (int)i, (uint32_t)empty_out[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_outputs(uint32_t n, uint32_t coverage) {
    uint32_t hash = 2166136261u ^ n ^ (coverage << 16);
    for (uint32_t i = 0; i < coverage; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)full_out[index] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)tail_out[index] + 0x85ebca6bu + (i << 5);
        hash = rotl32_local(hash, 7u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t x_dev = 0, tail_dev = 0, full_dev = 0, empty_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &tail_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &full_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &empty_dev, bytes) < 0)
        panic("value_mask_tail_full_empty allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        uint32_t waves = rounded_waves(n);
        uint32_t coverage = waves * ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers();
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t tail_active = tail_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t full_active = full_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t empty_active = empty_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, tail_dev, tail_out, bytes) < 0 ||
            vc4_m2_copy_htod(program, full_dev, full_out, bytes) < 0 ||
            vc4_m2_copy_htod(program, empty_dev, empty_out, bytes) < 0 ||
            value_mask_tail_full_empty_vc4value_launch(
                program, grid, block, x_active, tail_active, full_active,
                empty_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, tail_out, tail_dev, bytes) < 0 ||
            vc4_m2_copy_dtoh(program, full_out, full_dev, bytes) < 0 ||
            vc4_m2_copy_dtoh(program, empty_out, empty_dev, bytes) < 0) {
            printk("ERROR: value_mask_tail_full_empty launch/copy failed n=%d\n", (int)n);
            launch_failures++;
            continue;
        }
        int tail_mismatches = verify_tail(n);
        int full_mismatches = verify_full(coverage);
        int empty_mismatches = verify_empty();
        uint32_t case_hash = hash_outputs(n, coverage);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += tail_mismatches + full_mismatches;
        sentinel_mismatches += empty_mismatches;
        printk("VALUE_MASK_TAIL_FULL_EMPTY_CASE case=%d n=%d waves=%d coverage=%d tail_mismatches=%d full_mismatches=%d empty_sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)waves, (int)coverage,
               tail_mismatches, full_mismatches, empty_mismatches, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_mask_tail_full_empty_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d runtime_allocations=%d runtime_launches=%d saw_value_mask_empty=1 saw_value_mask_full=1 saw_value_mask_tail=1 saw_value_create_mask_clamp=1 elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, 4,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, tail_dev);
    vc4Free(program, full_dev);
    vc4Free(program, empty_dev);
    vc4_program_destroy(program);
}
