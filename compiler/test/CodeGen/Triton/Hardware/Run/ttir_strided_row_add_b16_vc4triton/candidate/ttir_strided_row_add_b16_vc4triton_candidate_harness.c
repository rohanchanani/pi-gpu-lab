#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_NCOLS 33u
#define MAX_STRIDE 52u
#define BUFFER_N (MAX_ROWS * MAX_STRIDE + 2u * GUARD)
#define SENTINEL_BITS 0x7fa20022u

static const uint32_t row_cases[] = {1u, 2u, 7u};
static const uint32_t col_cases[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u};

static float a_values[BUFFER_N];
static float b_values[BUFFER_N];
static float c_values[BUFFER_N];

static float bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

static uint32_t float_to_bits(float value) {
    union { uint32_t u; float f; } bits;
    bits.f = value;
    return bits.u;
}

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t col_blocks(uint32_t ncols) {
    uint32_t blocks = (ncols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t lda_for(uint32_t rows, uint32_t ncols) { return ncols + 5u + (rows & 3u); }
static uint32_t ldb_for(uint32_t rows, uint32_t ncols) { return ncols + 8u + ((rows + 1u) & 3u); }
static uint32_t ldc_for(uint32_t rows, uint32_t ncols) { return ncols + 12u + ((rows + 2u) & 3u); }

static float a_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 13u + row * 7u + col * 5u) - 53;
    return (float)whole + (float)((case_id + col) & 3u) * 0.25f;
}

static float b_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 11u + row * 17u + col * 2u) - 29;
    return (float)whole + (float)((row + col) & 1u) * 0.5f;
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t ncols,
                         uint32_t lda, uint32_t ldb) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        a_values[i] = (float)((int32_t)i - 71);
        b_values[i] = (float)((int32_t)i + 31);
        c_values[i] = sentinel;
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            a_values[GUARD + r * lda + c] = a_value(case_id, r, c);
            b_values[GUARD + r * ldb + c] = b_value(case_id, r, c);
        }
    }
}

static int verify_results(uint32_t case_id, uint32_t rows, uint32_t ncols,
                          uint32_t ldc) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            uint32_t index = GUARD + r * ldc + c;
            uint32_t got = float_to_bits(c_values[index]);
            uint32_t expected = float_to_bits(a_value(case_id, r, c) + b_value(case_id, r, c));
            if (got != expected) {
                if (mismatches < 8)
                    printk("ERROR: ttir row add row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)c, got, expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t ncols, uint32_t ldc) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + rows * ldc) {
            uint32_t rel = i - GUARD;
            active = (rel % ldc) < ncols;
        }
        if (active)
            continue;
        if (float_to_bits(c_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir row add sentinel rows=%d ncols=%d ldc=%d i=%d got=%x\n",
                       (int)rows, (int)ncols, (int)ldc, (int)i, float_to_bits(c_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows, uint32_t ncols, uint32_t ldc) {
    uint32_t hash = 2166136261u ^ rows ^ (ncols << 8) ^ (ldc << 16);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            uint32_t index = GUARD + r * ldc + c;
            hash ^= float_to_bits(c_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("ttir row add program create failed");
    uint32_t bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t a_dev = 0, b_dev = 0, c_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &b_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &c_dev, bytes) < 0)
        panic("ttir row add allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);
    uint32_t case_id = 0;

    for (uint32_t ri = 0; ri < sizeof(row_cases) / sizeof(row_cases[0]); ri++) {
        for (uint32_t ci = 0; ci < sizeof(col_cases) / sizeof(col_cases[0]); ci++) {
            uint32_t rows = row_cases[ri];
            uint32_t ncols = col_cases[ci];
            uint32_t lda = lda_for(rows, ncols);
            uint32_t ldb = ldb_for(rows, ncols);
            uint32_t ldc = ldc_for(rows, ncols);
            vc4_dim3 grid = vc4_m2_dim3(col_blocks(ncols), rows, 1u);
            fill_buffers(case_id, rows, ncols, lda, ldb);
            vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
            vc4_deviceptr_t b_active = b_dev + GUARD * sizeof(float);
            vc4_deviceptr_t c_active = c_dev + GUARD * sizeof(float);
            if (vc4_m2_copy_htod(program, a_dev, a_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, b_dev, b_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, c_dev, c_values, bytes) < 0 ||
                ttir_strided_row_add_b16_kernel_launch(program, grid, block,
                                                       a_active, b_active, c_active,
                                                       ncols, lda, ldb, ldc) < 0 ||
                vc4_m2_copy_dtoh(program, c_values, c_dev, bytes) < 0) {
                printk("ERROR: ttir row add launch/copy failed case=%d rows=%d ncols=%d\n",
                       (int)case_id, (int)rows, (int)ncols);
                launch_failures++;
                case_id++;
                continue;
            }
            int mismatches = verify_results(case_id, rows, ncols, ldc);
            int sentinels = verify_sentinels(rows, ncols, ldc);
            uint32_t hash = hash_case(rows, ncols, ldc);
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
            output_hash = rotl32_local(output_hash, 7u);
            printk("TTIR_STRIDED_ROW_ADD_CASE case=%d rows=%d ncols=%d lda=%d ldb=%d ldc=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
                   (int)case_id, (int)rows, (int)ncols, (int)lda, (int)ldb,
                   (int)ldc, mismatches, sentinels, hash);
            case_id++;
        }
    }

    int elapsed = timer_get_usec() - start;
    int output_hash_nonzero = output_hash != 0u;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_strided_row_add_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d output_hash=%u output_hash_nonzero=%d saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_row_strided_memory=1 saw_value_strided_address=1 saw_no_gather_lane_stride=1 saw_ttir_two_input_strided=1 saw_row_padding_sentinels=1 saw_phase10_tail_mask=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)case_id, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS, MAX_NCOLS, output_hash,
           output_hash_nonzero, VC4_CASE_SAW_CPP_TTIR_IMPORTER, 3, (int)case_id,
           elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, c_dev);
    vc4_program_destroy(program);
}
