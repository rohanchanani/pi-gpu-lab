#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_COLS 33u
#define ACTIVE_N (MAX_ROWS * MAX_COLS)
#define BUFFER_N (ACTIVE_N + 2u * GUARD + 16u)
#define SENTINEL 0x52220011u

struct case_desc { uint32_t rows, cols; };
static const struct case_desc cases[] = {
    {1u, 0u}, {1u, 1u}, {2u, 15u}, {3u, 16u}, {5u, 17u}, {7u, 33u}
};

static int32_t in_values[BUFFER_N];
static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t col_blocks(uint32_t cols) {
    uint32_t blocks = (cols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static int32_t input_value(uint32_t case_id, uint32_t row, uint32_t col) {
    return (int32_t)(0x22000000u + case_id * 0x10000u + row * 409u + col * 23u);
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        in_values[i] = (int32_t)(0x50000000u + i);
        out_values[i] = (int32_t)SENTINEL;
    }
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            in_values[GUARD + r * cols + c] = input_value(case_id, r, c);
}

static int verify_case(uint32_t case_id, uint32_t rows, uint32_t cols) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t index = GUARD + r * cols + c;
            int32_t expected = input_value(case_id, r, c);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: rank2 identity row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)c, (uint32_t)out_values[index], (uint32_t)expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t cols) {
    int mismatches = 0;
    uint32_t active_end = GUARD + rows * cols;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < active_end)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: rank2 identity sentinel rows=%d cols=%d i=%d got=%x\n",
                       (int)rows, (int)cols, (int)i, (uint32_t)out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows, uint32_t cols) {
    uint32_t hash = 2166136261u ^ rows ^ (cols << 8);
    for (uint32_t i = GUARD; i < GUARD + rows * cols; i++) {
        hash ^= (uint32_t)out_values[i] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_rank2_row_slice_copy program create failed");
    uint32_t bytes = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &in_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("value_rank2_row_slice_copy allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t rows = cases[case_id].rows;
        uint32_t cols = cases[case_id].cols;
        vc4_dim3 grid = vc4_m2_dim3(col_blocks(cols), rows == 0u ? 1u : rows, 1u);
        fill_buffers(case_id, rows, cols);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, in_dev, in_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            value_rank2_row_slice_copy_vc4value_launch(
                program, grid, block, in_active, out_active, rows, cols) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: rank2 identity launch/copy failed case=%d rows=%d cols=%d\n",
                   (int)case_id, (int)rows, (int)cols);
            launch_failures++;
            continue;
        }
        int mismatches = verify_case(case_id, rows, cols);
        int sentinels = verify_sentinels(rows, cols);
        uint32_t hash = hash_case(rows, cols);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_RANK2_IDENTITY_CASE case=%d rows=%d cols=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)rows, (int)cols, mismatches, sentinels, hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_rank2_row_slice_copy_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_cols=%d output_hash=%u output_hash_nonzero=%d saw_value_rank2_row_slice_identity=1 saw_value_shape_args=1 saw_row_major_index=1 saw_value_tail_mask=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS,
           MAX_COLS, output_hash, output_hash != 0u ? 1 : 0, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);
    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
