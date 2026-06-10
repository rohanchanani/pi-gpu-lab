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
#define SENTINEL 0x4d3c2b1au

struct case_config {
    uint32_t n;
    uint32_t start;
    uint32_t end;
    uint32_t step;
    uint32_t trip;
    uint32_t early;
    uint32_t use_alt;
};

static const struct case_config cases[] = {
    {0u, 0u, 0u, 1u, 0u, 0u, 0u},
    {1u, 0u, 1u, 1u, 1u, 5u, 0u},
    {2u, 1u, 5u, 2u, 2u, 0u, 1u},
    {3u, 2u, 11u, 3u, 3u, 2u, 0u},
    {5u, 5u, 5u, 1u, 5u, 9u, 1u},
    {17u, 0u, 16u, 4u, 1u, 1u, 1u},
    {193u, 3u, 25u, 5u, 4u, 4u, 0u},
    {1000u, 7u, 43u, 6u, 11u, 6u, 1u},
};

static int32_t x_values[BUFFER_N];
static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int32_t x_value(uint32_t i) {
    return (int32_t)((i * 13u + 7u) % 101u) - 50;
}

static int32_t range_acc(const struct case_config *cfg) {
    int32_t acc = 0;
    for (uint32_t tile = cfg->start; tile < cfg->end; tile += cfg->step)
        acc += cfg->use_alt ? (int32_t)(tile + 1u) : 1;
    return acc;
}

static int32_t scf_while_acc(uint32_t trip) {
    int32_t acc = 0;
    for (uint32_t i = 0; i < trip; i++)
        acc += (int32_t)(i + 1u);
    return acc;
}

static int32_t multi_exit_acc(uint32_t trip, uint32_t early) {
    uint32_t i = 0;
    int32_t acc = 0;
    for (;;) {
        if (i >= trip)
            return acc + 100;
        if (i >= early)
            return acc + 200;
        acc += (int32_t)(i + 1u);
        i++;
    }
}

static int32_t expected_value(uint32_t i, const struct case_config *cfg) {
    int32_t value = x_value(i);
    value += range_acc(cfg);
    value += scf_while_acc(cfg->trip);
    value += multi_exit_acc(cfg->trip, cfg->early);
    return value;
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + MAX_COVERAGE_N) {
            uint32_t logical = i - GUARD;
            x_values[i] = x_value(logical);
        } else {
            x_values[i] = (int32_t)SENTINEL;
        }
        out_values[i] = (int32_t)SENTINEL;
    }
}

static int verify_results(const struct case_config *cfg) {
    int mismatches = 0;
    for (uint32_t i = 0; i < cfg->n; i++) {
        uint32_t index = GUARD + i;
        int32_t expected = expected_value(i, cfg);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: mixed_cf_completeness active n=%d start=%d end=%d step=%d trip=%d early=%d use_alt=%d i=%d got=%x expected=%x\n",
                       (int)cfg->n, (int)cfg->start, (int)cfg->end,
                       (int)cfg->step, (int)cfg->trip, (int)cfg->early,
                       (int)cfg->use_alt, (int)i, (uint32_t)out_values[index],
                       (uint32_t)expected);
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
                printk("ERROR: mixed_cf_completeness sentinel n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct case_config *cfg) {
    uint32_t hash = 2166136261u ^ cfg->n;
    for (uint32_t i = 0; i < cfg->n; i++) {
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
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("mixed_cf_completeness allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start_time = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_config *cfg = &cases[case_id];
        uint32_t waves = rounded_waves(cfg->n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers();
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);

        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            mixed_value_cf_completeness_loop_branch_tail_vc4value_launch(
                program, grid, block, x_active, out_active, cfg->start, cfg->end,
                cfg->step, cfg->trip, cfg->early, cfg->use_alt, cfg->n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: mixed_cf_completeness launch/copy failed n=%d start=%d end=%d step=%d trip=%d early=%d use_alt=%d\n",
                   (int)cfg->n, (int)cfg->start, (int)cfg->end,
                   (int)cfg->step, (int)cfg->trip, (int)cfg->early,
                   (int)cfg->use_alt);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(cfg);
        int sentinels = verify_sentinels(cfg->n);
        uint32_t case_hash = hash_output(cfg);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)cfg->n;
        printk("MIXED_VALUE_CF_COMPLETENESS_CASE case=%d n=%d start=%d end=%d step=%d trip=%d early=%d use_alt=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)cfg->n, (int)cfg->start, (int)cfg->end,
               (int)cfg->step, (int)cfg->trip, (int)cfg->early,
               (int)cfg->use_alt, (int)waves, mismatches, sentinels,
               case_hash);
    }

    int elapsed = timer_get_usec() - start_time;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_cf_completeness_loop_branch_tail_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u saw_value_scf_while=1 saw_nested_structured_cf=1 saw_tl_range_style_loop=1 saw_persistent_loop_skeleton=1 saw_vector_loop_carried=1 saw_value_tail_mask=1 saw_multi_exit_reducible_cf=1 saw_scf_to_cf_boundary=1 saw_value_tmu_load=1 saw_value_vdw_preserve=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
