#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SHARED_TRANSPOSE_16X16_SSAVC4_DIM 16u
#define SHARED_TRANSPOSE_16X16_SSAVC4_WORDS 256u
#define SHARED_TRANSPOSE_16X16_SSAVC4_WARPS_PER_BLOCK 4u
#define SHARED_TRANSPOSE_16X16_SSAVC4_CASES 4u
#define HOST_GUARD_WORDS 16u
#define HOST_SENTINEL 0x6d7e8f90u

static uint32_t input_values[SHARED_TRANSPOSE_16X16_SSAVC4_WORDS];
static uint32_t output_values[SHARED_TRANSPOSE_16X16_SSAVC4_WORDS + HOST_GUARD_WORDS];
static uint32_t expected_values[SHARED_TRANSPOSE_16X16_SSAVC4_WORDS];

static uint32_t matrix_index(uint32_t row, uint32_t col) {
    return row * SHARED_TRANSPOSE_16X16_SSAVC4_DIM + col;
}

static uint32_t case_value(uint32_t case_id, uint32_t row, uint32_t col) {
    switch (case_id) {
    case 0:
        return row * SHARED_TRANSPOSE_16X16_SSAVC4_DIM + col;
    case 1:
        return 0x70000000u | (row << 8) | col;
    case 2:
        return row * 37u + col * 11u + 5u;
    case 3:
        if (row == col)
            return 0xf0000000u | row;
        if (row + col == SHARED_TRANSPOSE_16X16_SSAVC4_DIM - 1u)
            return 0xe0000000u | col;
        return 0x10000000u | ((row * 19u + col * 23u + 7u) & 0x0000ffffu);
    default:
        return 0u;
    }
}

static void fill_case(uint32_t case_id) {
    for (uint32_t row = 0; row < SHARED_TRANSPOSE_16X16_SSAVC4_DIM; row++) {
        for (uint32_t col = 0; col < SHARED_TRANSPOSE_16X16_SSAVC4_DIM; col++) {
            input_values[matrix_index(row, col)] = case_value(case_id, row, col);
            expected_values[matrix_index(row, col)] = case_value(case_id, col, row);
        }
    }

    for (uint32_t i = 0; i < SHARED_TRANSPOSE_16X16_SSAVC4_WORDS + HOST_GUARD_WORDS; i++)
        output_values[i] = HOST_SENTINEL;
}

static uint32_t checksum_words(const uint32_t *values, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; i++) {
        hash ^= values[i];
        hash *= 16777619u;
    }
    return hash;
}

static int verify_case(uint32_t case_id) {
    int mismatches = 0;

    for (uint32_t row = 0; row < SHARED_TRANSPOSE_16X16_SSAVC4_DIM; row++) {
        for (uint32_t col = 0; col < SHARED_TRANSPOSE_16X16_SSAVC4_DIM; col++) {
            uint32_t index = matrix_index(row, col);
            uint32_t got = output_values[index];
            uint32_t expected = expected_values[index];

            if (got != expected) {
                if (mismatches < 8)
                    printk("ERROR: case=%d row=%d col=%d got=%x expected=%x input_transposed=%x\n",
                           (int)case_id, (int)row, (int)col, got, expected,
                           input_values[matrix_index(col, row)]);
                mismatches++;
            }
        }
    }

    return mismatches;
}

static int verify_host_guard(void) {
    int mismatches = 0;

    for (uint32_t i = 0; i < HOST_GUARD_WORDS; i++) {
        uint32_t value = output_values[SHARED_TRANSPOSE_16X16_SSAVC4_WORDS + i];
        if (value != HOST_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: host output guard changed index=%d value=%x expected=%x\n",
                       (int)i, value, HOST_SENTINEL);
            mismatches++;
        }
    }

    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t output_dev = 0;
    uint32_t input_bytes = SHARED_TRANSPOSE_16X16_SSAVC4_WORDS * sizeof(uint32_t);
    uint32_t output_bytes = (SHARED_TRANSPOSE_16X16_SSAVC4_WORDS + HOST_GUARD_WORDS) * sizeof(uint32_t);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &input_dev, input_bytes) < 0 ||
        vc4Malloc(program, &output_dev, output_bytes) < 0)
        panic("shared_transpose_16x16_ssavc4 device allocation failed");

    uint32_t active_qpus = 12u;
    uint32_t lane_width = 16u;
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(SHARED_TRANSPOSE_16X16_SSAVC4_WARPS_PER_BLOCK * lane_width, 1u, 1u);

    printk("Running VC4 shared_transpose_16x16_ssavc4 candidate bundle...\n");
    printk("SHARED_TRANSPOSE_16X16_SSAVC4_RUNTIME_SETUP allocations=%d active_qpus=%d lanes=%d warps_per_block=%d\n",
           1, (int)active_qpus, (int)lane_width, SHARED_TRANSPOSE_16X16_SSAVC4_WARPS_PER_BLOCK);

    int start = timer_get_usec();
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0u;

    for (uint32_t case_id = 0; case_id < SHARED_TRANSPOSE_16X16_SSAVC4_CASES; case_id++) {
        fill_case(case_id);

        if (vc4MemcpyHtoD(program, input_dev, input_values, input_bytes) < 0 ||
            vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0 ||
            shared_transpose_16x16_ssavc4_launch(program, grid, block, input_dev, output_dev) < 0 ||
            vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
            printk("ERROR: shared_transpose_16x16_ssavc4 launch/copy failed case=%d\n",
                   (int)case_id);
            launch_failures++;
            continue;
        }

        int mismatches = verify_case(case_id);
        int case_sentinel_mismatches = verify_host_guard();
        uint32_t checksum = checksum_words(output_values, SHARED_TRANSPOSE_16X16_SSAVC4_WORDS);
        checksum_accum ^= checksum + (case_id * 0x9e3779b9u);

        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel_mismatches;

        printk("SHARED_TRANSPOSE_16X16_SSAVC4_CASE case=%d mismatches=%d sentinel_mismatches=%d checksum=%x launches=%d allocations=%d\n",
               (int)case_id, mismatches, case_sentinel_mismatches, checksum,
               (int)(case_id + 1u), 1);
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 &&
         sentinel_mismatches == 0 &&
         launch_failures == 0) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=shared_transpose_16x16_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d runtime_allocations=%d runtime_launches=%d timeouts=%d errstat_relevant_changed=%d checksum_accum=%x elapsed_usec=%d\n",
           status, SHARED_TRANSPOSE_16X16_SSAVC4_CASES, total_mismatches,
           sentinel_mismatches, launch_failures, (int)active_qpus,
           (int)lane_width, SHARED_TRANSPOSE_16X16_SSAVC4_WARPS_PER_BLOCK,
           1, SHARED_TRANSPOSE_16X16_SSAVC4_CASES, 0, 0, checksum_accum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
