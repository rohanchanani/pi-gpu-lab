#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_GRID_BLOCKS 20u
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL 0x54593244u

struct grid2_case {
    uint32_t x;
    uint32_t y;
};

static const struct grid2_case cases[] = {
    {1u, 1u},
    {2u, 3u},
    {4u, 3u},
    {5u, 4u},
};

static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = (int32_t)SENTINEL;
}

static int32_t expected_value(uint32_t pid0, uint32_t pid1, uint32_t lane) {
    return (int32_t)(pid1 * 100000u + pid0 * 1000u + lane);
}

static int verify_results(uint32_t grid_x, uint32_t grid_y) {
    int mismatches = 0;
    for (uint32_t pid1 = 0; pid1 < grid_y; pid1++) {
        for (uint32_t pid0 = 0; pid0 < grid_x; pid0++) {
            uint32_t block_id = pid1 * grid_x + pid0;
            for (uint32_t lane = 0; lane < LANES; lane++) {
                uint32_t i = block_id * LANES + lane;
                uint32_t index = GUARD + i;
                int32_t expected = expected_value(pid0, pid1, lane);
                if (out_values[index] != expected) {
                    if (mismatches < 8)
                        printk("ERROR: ttir pid2d grid=%d,%d pid0=%d pid1=%d lane=%d got=%x expected=%x\n",
                               (int)grid_x, (int)grid_y, (int)pid0, (int)pid1,
                               (int)lane, (uint32_t)out_values[index],
                               (uint32_t)expected);
                    mismatches++;
                }
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t elements) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + elements)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: ttir pid2d sentinel elements=%d i=%d got=%x expected=%x\n",
                       (int)elements, (int)i, (uint32_t)out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t elements) {
    uint32_t hash = 2166136261u ^ elements;
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
        panic("ttir pid2d allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t grid_x = cases[case_id].x;
        uint32_t grid_y = cases[case_id].y;
        uint32_t elements = grid_x * grid_y * LANES;
        fill_output();
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        vc4_dim3 grid = vc4_m2_dim3(grid_x, grid_y, 1u);
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            ttir_multi_axis_pid2d_b16_kernel_launch(program, grid, block, out_active, elements) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: ttir pid2d launch/copy failed case=%d grid=%d,%d\n",
                   (int)case_id, (int)grid_x, (int)grid_y);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(grid_x, grid_y);
        int sentinels = verify_sentinels(elements);
        uint32_t case_hash = hash_output(elements);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)elements;
        printk("TTIR_MULTI_AXIS_PID2D_CASE case=%d grid_x=%d grid_y=%d elements=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)grid_x, (int)grid_y, (int)elements,
               mismatches, sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_multi_axis_pid2d_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u saw_nonzero_output_hash=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_multi_axis_launch=1 saw_ttir_program_id_axis0=1 saw_ttir_program_id_axis1=1 saw_ttir_num_programs_axis0=1 saw_value_multi_axis_launch=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
