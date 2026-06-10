#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_NCOLS 33u
#define MAX_LDA (MAX_NCOLS + 7u)
#define ACTIVE_N (MAX_ROWS * MAX_LDA)
#define BUFFER_N (ACTIVE_N + 2u * GUARD)
#define SENTINEL 0x51d1d001u

static const uint32_t row_cases[] = {1u, 2u, 7u};
static const uint32_t col_cases[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u};

static int32_t in_values[BUFFER_N];
static int32_t out_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t col_blocks(uint32_t ncols) {
    uint32_t blocks = (ncols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t lda_for(uint32_t rows, uint32_t ncols) {
    return ncols + 5u + (rows & 3u);
}

static int32_t input_value(uint32_t case_id, uint32_t row, uint32_t col) {
    return (int32_t)(0x11000000u + case_id * 0x10000u + row * 257u + col * 17u);
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t ncols, uint32_t lda) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        in_values[i] = (int32_t)(0x40000000u + i);
        out_values[i] = (int32_t)SENTINEL;
    }
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < ncols; c++)
            in_values[GUARD + r * lda + c] = input_value(case_id, r, c);
}

static int verify_case(uint32_t case_id, uint32_t rows, uint32_t ncols, uint32_t lda) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            uint32_t index = GUARD + r * lda + c;
            int32_t expected = input_value(case_id, r, c);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: rank1 stride row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)c, (uint32_t)out_values[index], (uint32_t)expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t ncols, uint32_t lda) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + rows * lda) {
            uint32_t rel = i - GUARD;
            uint32_t col = rel % lda;
            active = col < ncols;
        }
        if (active)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: rank1 stride sentinel rows=%d ncols=%d lda=%d i=%d got=%x\n",
                       (int)rows, (int)ncols, (int)lda, (int)i, (uint32_t)out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows, uint32_t ncols, uint32_t lda) {
    uint32_t hash = 2166136261u ^ rows ^ (ncols << 8) ^ (lda << 16);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            uint32_t index = GUARD + r * lda + c;
            hash ^= (uint32_t)out_values[index] + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_strided_rank1_row_copy program create failed");
    uint32_t bytes = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &in_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("value_strided_rank1_row_copy allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);
    uint32_t case_id = 0;

    for (uint32_t ri = 0; ri < sizeof(row_cases) / sizeof(row_cases[0]); ri++) {
        for (uint32_t ci = 0; ci < sizeof(col_cases) / sizeof(col_cases[0]); ci++) {
            uint32_t rows = row_cases[ri];
            uint32_t ncols = col_cases[ci];
            uint32_t lda = lda_for(rows, ncols);
            uint32_t total = rows * lda;
            vc4_dim3 grid = vc4_m2_dim3(col_blocks(ncols), rows, 1u);
            fill_buffers(case_id, rows, ncols, lda);
            vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(int32_t);
            vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
            if (vc4_m2_copy_htod(program, in_dev, in_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
                value_strided_rank1_row_copy_vc4value_launch(
                    program, grid, block, in_active, out_active, total, ncols, lda) < 0 ||
                vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
                printk("ERROR: rank1 stride launch/copy failed case=%d rows=%d ncols=%d lda=%d\n",
                       (int)case_id, (int)rows, (int)ncols, (int)lda);
                launch_failures++;
                case_id++;
                continue;
            }
            int mismatches = verify_case(case_id, rows, ncols, lda);
            int sentinels = verify_sentinels(rows, ncols, lda);
            uint32_t hash = hash_case(rows, ncols, lda);
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
            output_hash = rotl32_local(output_hash, 7u);
            printk("VALUE_RANK1_STRIDED_CASE case=%d rows=%d ncols=%d lda=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
                   (int)case_id, (int)rows, (int)ncols, (int)lda, mismatches, sentinels, hash);
            case_id++;
        }
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_strided_rank1_row_copy_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d output_hash=%u output_hash_nonzero=%d saw_value_rank1_flattened_stride=1 saw_value_row_stride=1 saw_row_padding_sentinels=1 saw_value_tail_mask=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)case_id, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS, MAX_NCOLS, output_hash,
           output_hash != 0u ? 1 : 0, 2, (int)case_id, elapsed);
    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
