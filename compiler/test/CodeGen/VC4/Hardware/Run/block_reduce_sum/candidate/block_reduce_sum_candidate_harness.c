#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define BLOCK_REDUCE_SUM_EPSILON 0.0005f
#define CHECKSUM_SCALE 4096.0f
#define BLOCK_REDUCE_SUM_LANE_WIDTH 16u
#define BLOCK_REDUCE_SUM_MAX_QPUS 12u
#define BLOCK_REDUCE_SUM_RESULT_LANES 16u
#define BLOCK_REDUCE_SUM_SENTINEL (-12345.0f)

enum {
    BRS_CASES = 6,
    BRS_MAX_BLOCKS = 3,
    BRS_MAX_VALUES_PER_BLOCK = 192,
    BRS_MAX_INPUT = BRS_MAX_BLOCKS * BRS_MAX_VALUES_PER_BLOCK,
    BRS_MAX_OUTPUT = BRS_MAX_BLOCKS * BLOCK_REDUCE_SUM_RESULT_LANES
};

struct block_reduce_sum_case {
    uint32_t blocks;
    uint32_t warps_per_block;
    uint32_t values_per_block;
};

static const struct block_reduce_sum_case cases[BRS_CASES] = {
    {1u, 1u, 1u},
    {1u, 2u, 31u},
    {1u, 4u, 64u},
    {1u, 12u, 192u},
    {1u, 12u, 173u},
    {3u, 4u, 59u},
};

static float input_values[BRS_MAX_INPUT];
static float out_values[BRS_MAX_OUTPUT];
static float expected_values[BRS_MAX_OUTPUT];

static float absf_local(float value) {
    return value < 0.0f ? -value : value;
}

static float make_input_value(uint32_t i, uint32_t case_id) {
    int raw = (int)((i * 5u + case_id * 7u + 3u) % 17u) - 8;
    return (float)raw * 0.125f;
}

static void fill_inputs(uint32_t case_id, uint32_t blocks, uint32_t values_per_block) {
    for (uint32_t i = 0; i < BRS_MAX_INPUT; i++)
        input_values[i] = 0.0f;
    for (uint32_t i = 0; i < BRS_MAX_OUTPUT; i++) {
        out_values[i] = BLOCK_REDUCE_SUM_SENTINEL;
        expected_values[i] = BLOCK_REDUCE_SUM_SENTINEL;
    }

    for (uint32_t block = 0; block < blocks; block++) {
        for (uint32_t i = 0; i < values_per_block; i++) {
            uint32_t logical = block * values_per_block + i;
            input_values[logical] = make_input_value(logical, case_id);
        }
    }
}

static void run_cpu_reference(uint32_t blocks, uint32_t values_per_block) {
    for (uint32_t block = 0; block < blocks; block++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < values_per_block; i++)
            sum += input_values[block * values_per_block + i];

        for (uint32_t lane = 0; lane < BLOCK_REDUCE_SUM_RESULT_LANES; lane++)
            expected_values[block * BLOCK_REDUCE_SUM_RESULT_LANES + lane] = sum;
    }
}

static void verify_results(uint32_t case_id, uint32_t blocks, int *mismatches, float *max_abs_diff) {
    *mismatches = 0;
    *max_abs_diff = 0.0f;

    uint32_t count = blocks * BLOCK_REDUCE_SUM_RESULT_LANES;
    for (uint32_t i = 0; i < count; i++) {
        float diff = out_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > BLOCK_REDUCE_SUM_EPSILON) {
            if (*mismatches < 8) {
                uint32_t block = i / BLOCK_REDUCE_SUM_RESULT_LANES;
                uint32_t lane = i % BLOCK_REDUCE_SUM_RESULT_LANES;
                printk("BLOCK_REDUCE_BLOCK case=%d block=%d lane=%d gpu=%f cpu=%f diff=%f\n",
                       (int)case_id, (int)block, (int)lane, out_values[i],
                       expected_values[i], diff);
            }
            (*mismatches)++;
        }
    }
}

static int scaled_checksum(const float *values, uint32_t count) {
    int checksum = 0;
    for (uint32_t i = 0; i < count; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static int launch_case(struct vc4_program *program,
                       vc4_deviceptr_t input_dev,
                       vc4_deviceptr_t out_dev,
                       uint32_t blocks,
                       uint32_t values_per_block,
                       uint32_t warps_per_block) {
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block_dim = vc4_m2_dim3(warps_per_block * BLOCK_REDUCE_SUM_LANE_WIDTH, 1u, 1u);

    for (uint32_t block = 0; block < blocks; block++) {
        vc4_deviceptr_t block_input = input_dev + block * values_per_block * sizeof(float);
        vc4_deviceptr_t block_out = out_dev + block * BLOCK_REDUCE_SUM_RESULT_LANES * sizeof(float);

        if (block_reduce_sum_launch(program, grid, block_dim, block_input, block_out,
                                    values_per_block, 0u, warps_per_block, 0u) < 0)
            return -1;
    }

    return 0;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t input_bytes = BRS_MAX_INPUT * sizeof(float);
    uint32_t out_bytes = BRS_MAX_OUTPUT * sizeof(float);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &input_dev, input_bytes) < 0 ||
        vc4Malloc(program, &out_dev, out_bytes) < 0)
        panic("block_reduce_sum device allocation failed");

    uint32_t active_qpus = BLOCK_REDUCE_SUM_MAX_QPUS;
    uint32_t lane_width = BLOCK_REDUCE_SUM_LANE_WIDTH;

    printk("BLOCK_REDUCE_SUM_RUNTIME_SETUP max_blocks=%d max_values_per_block=%d allocations=%d ident1=%x vpmbase=%x\n",
           BRS_MAX_BLOCKS, BRS_MAX_VALUES_PER_BLOCK, 1, 0u, 0u);

    int total_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t full_block_pass = 0u;
    uint32_t tail_pass = 0u;
    uint32_t multi_block_pass = 0u;
    uint32_t tile_waves = 0u;
    int start = timer_get_usec();

    for (uint32_t case_index = 0; case_index < BRS_CASES; case_index++) {
        uint32_t blocks = cases[case_index].blocks;
        uint32_t warps_per_block = cases[case_index].warps_per_block;
        uint32_t values_per_block = cases[case_index].values_per_block;

        fill_inputs(case_index, blocks, values_per_block);
        run_cpu_reference(blocks, values_per_block);

        if (vc4MemcpyHtoD(program, input_dev, input_values, input_bytes) < 0 ||
            vc4MemcpyHtoD(program, out_dev, out_values, out_bytes) < 0 ||
            launch_case(program, input_dev, out_dev, blocks, values_per_block, warps_per_block) < 0 ||
            vc4MemcpyDtoH(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: block_reduce_sum launch/copy failed case=%d blocks=%d warps_per_block=%d values_per_block=%d\n",
                   (int)case_index, (int)blocks, (int)warps_per_block, (int)values_per_block);
            launch_failures++;
            continue;
        }

        tile_waves += blocks;

        int mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, blocks, &mismatches, &max_abs_diff);
        int checksum = scaled_checksum(out_values, blocks * BLOCK_REDUCE_SUM_RESULT_LANES);
        int expected_checksum = scaled_checksum(expected_values, blocks * BLOCK_REDUCE_SUM_RESULT_LANES);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n",
                   (int)case_index, checksum, expected_checksum);
            mismatches++;
        }

        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;

        total_mismatches += mismatches;
        checksum_accum += checksum;

        if (case_index == 3u && mismatches == 0)
            full_block_pass = 1u;
        if (case_index == 4u && mismatches == 0)
            tail_pass = 1u;
        if (case_index == 5u && mismatches == 0)
            multi_block_pass = 1u;

        printk("BLOCK_REDUCE_SUM_CASE case=%d blocks=%d warps_per_block=%d values_per_block=%d mismatches=%d checksum=%d max_abs_diff=%f launches=%d tile_waves=%d allocations=%d\n",
               (int)case_index, (int)blocks, (int)warps_per_block,
               (int)values_per_block, mismatches, checksum, max_abs_diff,
               (int)(case_index + 1u), (int)tile_waves, 1);
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 &&
         launch_failures == 0 &&
         full_block_pass == 1u &&
         tail_pass == 1u &&
         multi_block_pass == 1u &&
         max_abs_diff_overall <= BLOCK_REDUCE_SUM_EPSILON) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=block_reduce_sum status=%s cases=%d total_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_values_per_block=%d full_block_pass=%d tail_pass=%d multi_block_pass=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_tile_waves=%d timeouts=%d errstat_relevant_changed=%d srqcs_after_last_wave=%x elapsed_usec=%d\n",
           status, BRS_CASES, total_mismatches, launch_failures,
           (int)active_qpus, (int)lane_width, BRS_MAX_VALUES_PER_BLOCK,
           (int)full_block_pass, (int)tail_pass, (int)multi_block_pass,
           checksum_accum, max_abs_diff_overall, 1, BRS_CASES,
           (int)tile_waves, 0, 0, 0u, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
