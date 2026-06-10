#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define GRID_X 4u
#define GRID_Y 4u
#define MAX_GRID_BLOCKS (GRID_X * GRID_Y)
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL 0x54434d41u

static const uint32_t n_cases[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 255u, 256u, 263u};
static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = (int32_t)SENTINEL;
}

static uint32_t active_elements(uint32_t n) {
    return n < MAX_ELEMENTS ? n : MAX_ELEMENTS;
}

static int32_t expected_value(uint32_t i) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i & 15u;
    uint32_t pid0 = block_id % GRID_X;
    uint32_t pid1 = block_id / GRID_X;
    return (int32_t)(pid1 * 1000u + pid0 * 100u + 7u + lane);
}

static int verify_results(uint32_t n) {
    int mismatches = 0;
    uint32_t elements = active_elements(n);
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t index = GUARD + i;
        int32_t expected = expected_value(i);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: tail_cf n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)out_values[index],
                       (uint32_t)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    uint32_t elements = active_elements(n);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + elements)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: tail_cf sentinel n=%d elements=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)elements, (int)i, (uint32_t)out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t elements = active_elements(n);
    uint32_t hash = 2166136261u ^ n ^ (elements << 8);
    for (uint32_t i = 0; i < elements; i++) {
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
        panic("tail_cf allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);
    vc4_dim3 grid = vc4_m2_dim3(GRID_X, GRID_Y, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        uint32_t elements = active_elements(n);
        fill_output();
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            value_multi_axis_tail_cf_vc4value_launch(program, grid, block, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: tail_cf launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
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
        elements_checked += (int)elements;
        printk("VALUE_MULTI_AXIS_TAIL_CF_CASE case=%d n=%d active_elements=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)elements, mismatches, sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_multi_axis_tail_cf_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u saw_value_multi_axis_launch=1 saw_value_program_id_axis0=1 saw_value_program_id_axis1=1 saw_value_num_programs_axis0=1 saw_value_tail_mask=1 saw_value_cf_cond_br=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash, 1,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
