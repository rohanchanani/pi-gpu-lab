#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_CASES 6u
#define VC4_M 17u
#define VC4_N 13u
#define VC4_K 9u
#define VC4_MICRO 4u
#define VC4_M_TILES 5u
#define VC4_N_TILES 4u
#define VC4_K_TILES 3u
#define VC4_A_PACKED_WORDS (VC4_M_TILES * VC4_K_TILES * 16u)
#define VC4_B_PACKED_WORDS (VC4_K_TILES * VC4_N_TILES * 16u)
#define VC4_A_STORAGE_WORDS 288u
#define VC4_B_STORAGE_WORDS 240u
#define VC4_OUTPUT_WORDS (VC4_M * VC4_N)
#define VC4_OUTPUT_STORAGE_WORDS 384u
#define VC4_SENTINEL 0x5a5a5a5au
#define VC4_POISON 0x01010101u

static const uint32_t kAPads[VC4_CASES] = {0u, 3u, 9u, 17u, 21u, 31u};
static const uint32_t kBPads[VC4_CASES] = {1u, 5u, 11u, 19u, 23u, 37u};
static const uint32_t kOutPads[VC4_CASES] = {2u, 7u, 13u, 29u, 41u, 59u};

static int32_t positive_mod_i32(int32_t value, int32_t modulus) {
    int32_t r = value % modulus;
    return r < 0 ? r + modulus : r;
}

static int32_t a_value(uint32_t i, uint32_t k) {
    return positive_mod_i32((int32_t)(9u * i + 4u * k + 7u), 73) - 36;
}

static int32_t b_value(uint32_t k, uint32_t j) {
    return positive_mod_i32((int32_t)(6u * k + 41u) - (int32_t)(5u * j), 83) - 41;
}

static int32_t expected_c(uint32_t i, uint32_t j) {
    int32_t sum = 0;
    for (uint32_t k = 0; k < VC4_K; ++k)
        sum += a_value(i, k) * b_value(k, j);
    return sum;
}

static uint32_t a_pack_offset(uint32_t tile_i, uint32_t tile_k,
                              uint32_t local_i, uint32_t local_k) {
    return ((tile_i * VC4_K_TILES + tile_k) * 16u) + local_i * 4u + local_k;
}

static uint32_t b_pack_offset(uint32_t tile_k, uint32_t tile_j,
                              uint32_t local_k, uint32_t local_j) {
    return ((tile_k * VC4_N_TILES + tile_j) * 16u) + local_k * 4u + local_j;
}

static void fill_inputs(uint32_t *a, uint32_t a_pad, uint32_t *b, uint32_t b_pad,
                        uint32_t *out, uint32_t *got) {
    for (uint32_t i = 0; i < VC4_A_STORAGE_WORDS; ++i)
        a[i] = VC4_POISON;
    for (uint32_t i = 0; i < VC4_B_STORAGE_WORDS; ++i)
        b[i] = VC4_POISON;
    for (uint32_t i = 0; i < VC4_OUTPUT_STORAGE_WORDS; ++i) {
        out[i] = VC4_SENTINEL;
        got[i] = 0;
    }

    for (uint32_t tile_i = 0; tile_i < VC4_M_TILES; ++tile_i) {
        for (uint32_t tile_k = 0; tile_k < VC4_K_TILES; ++tile_k) {
            for (uint32_t local_i = 0; local_i < VC4_MICRO; ++local_i) {
                for (uint32_t local_k = 0; local_k < VC4_MICRO; ++local_k) {
                    uint32_t i = tile_i * VC4_MICRO + local_i;
                    uint32_t k = tile_k * VC4_MICRO + local_k;
                    if (i < VC4_M && k < VC4_K) {
                        a[a_pad + a_pack_offset(tile_i, tile_k, local_i, local_k)] =
                            (uint32_t)a_value(i, k);
                    }
                }
            }
        }
    }

    for (uint32_t tile_k = 0; tile_k < VC4_K_TILES; ++tile_k) {
        for (uint32_t tile_j = 0; tile_j < VC4_N_TILES; ++tile_j) {
            for (uint32_t local_k = 0; local_k < VC4_MICRO; ++local_k) {
                for (uint32_t local_j = 0; local_j < VC4_MICRO; ++local_j) {
                    uint32_t k = tile_k * VC4_MICRO + local_k;
                    uint32_t j = tile_j * VC4_MICRO + local_j;
                    if (k < VC4_K && j < VC4_N) {
                        b[b_pad + b_pack_offset(tile_k, tile_j, local_k, local_j)] =
                            (uint32_t)b_value(k, j);
                    }
                }
            }
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    uint32_t logical_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    uint32_t sample_c00 = 0, sample_c012 = 0, sample_c160 = 0;
    uint32_t sample_c1612 = 0, sample_c75 = 0;
    vc4_deviceptr_t d_a = 0, d_b = 0, d_out = 0;
    int failures = 0;

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 8192));
    if (!failures) failures += vc4_m2_malloc(program, &d_a, VC4_A_STORAGE_WORDS * sizeof(uint32_t)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_b, VC4_B_STORAGE_WORDS * sizeof(uint32_t)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_out, VC4_OUTPUT_STORAGE_WORDS * sizeof(uint32_t)) < 0;

    for (uint32_t c = 0; c < VC4_CASES; ++c) {
        uint32_t host_a[VC4_A_STORAGE_WORDS];
        uint32_t host_b[VC4_B_STORAGE_WORDS];
        uint32_t host_out[VC4_OUTPUT_STORAGE_WORDS];
        uint32_t got[VC4_OUTPUT_STORAGE_WORDS];
        uint32_t a_pad = kAPads[c];
        uint32_t b_pad = kBPads[c];
        uint32_t out_pad = kOutPads[c];

        fill_inputs(host_a, a_pad, host_b, b_pad, host_out, got);

        if (!failures) failures += vc4_m2_copy_htod(program, d_a, host_a, sizeof(host_a)) < 0;
        if (!failures) failures += vc4_m2_copy_htod(program, d_b, host_b, sizeof(host_b)) < 0;
        if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
        if (!failures) {
            failures += vc4_m2_check_rc(
                "cute_gemm_tail_17x13x9_vc4tile_launch",
                cute_gemm_tail_17x13x9_vc4tile_launch(
                    program, vc4_m2_dim3(20, 1, 1), vc4_m2_dim3(16, 1, 1),
                    d_out + out_pad * sizeof(uint32_t),
                    d_a + a_pad * sizeof(uint32_t),
                    d_b + b_pad * sizeof(uint32_t)));
        }
        if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

        for (uint32_t i = 0; i < VC4_OUTPUT_STORAGE_WORDS; ++i) {
            uint32_t active = (i >= out_pad && i < out_pad + VC4_OUTPUT_WORDS);
            uint32_t expected = VC4_SENTINEL;
            if (active) {
                uint32_t linear = i - out_pad;
                uint32_t row = linear / VC4_N;
                uint32_t col = linear % VC4_N;
                expected = (uint32_t)expected_c(row, col);
                checksum_expected += expected;
                checksum_actual += got[i];
                if (c == 0u) {
                    if (row == 0u && col == 0u) sample_c00 = got[i];
                    if (row == 0u && col == 12u) sample_c012 = got[i];
                    if (row == 16u && col == 0u) sample_c160 = got[i];
                    if (row == 16u && col == 12u) sample_c1612 = got[i];
                    if (row == 7u && col == 5u) sample_c75 = got[i];
                }
            }
            if (got[i] != expected) {
                if (active) {
                    if (logical_mismatches < 16u) {
                        uint32_t linear = i - out_pad;
                        printk("mismatch case=%u row=%u col=%u got=%d expected=%d\n",
                               c, linear / VC4_N, linear % VC4_N,
                               (int32_t)got[i], (int32_t)expected);
                    }
                    ++logical_mismatches;
                } else {
                    ++sentinel_mismatches;
                }
            }
        }
    }

    uint32_t launch_failures = cute_gemm_tail_17x13x9_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = cute_gemm_tail_17x13x9_vc4tile_runtime_launches();
    const char *status = (!failures && logical_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == VC4_CASES) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=cute_gemm_tail_17x13x9_vc4tile status=%s cases=%u logical_shape=17x13x9 tile_counts_m_n_k=5,4,3 logical_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u sample_c00=%d sample_c012=%d sample_c160=%d sample_c1612=%d sample_c75=%d\n",
           status, VC4_CASES, logical_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected,
           (int32_t)sample_c00, (int32_t)sample_c012, (int32_t)sample_c160,
           (int32_t)sample_c1612, (int32_t)sample_c75);
    printk("DONE!!!\n");
}
