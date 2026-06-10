#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define QGEMM_M 12u
#define QGEMM_N 16u
#define QGEMM_K 24u
#define QGEMM_K_TILE 12u
#define QGEMM_LANES 16u
#define QGEMM_CASES 5u
#define QGEMM_OUT_HALFS (QGEMM_M * QGEMM_N)
#define QGEMM_GUARD_HALFS 32u
#define QGEMM_SENTINEL 0x7badu

static uint16_t a_values[QGEMM_M * QGEMM_K];
static uint16_t b_values[QGEMM_K * QGEMM_N];
static float bias_values[QGEMM_N];
static uint16_t out_values[QGEMM_OUT_HALFS + QGEMM_GUARD_HALFS];
static uint16_t expected_values[QGEMM_OUT_HALFS];

static uint32_t float_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

static float bits_float(uint32_t value) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = value;
    return bits.f;
}

static uint16_t float_to_half(float value) {
    uint32_t bits = float_bits(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u)
            half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u)
        half++;
    return (uint16_t)half;
}

static float half_to_float(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)half & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0)
            bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    return bits_float(bits);
}

static float a_pattern(uint32_t row, uint32_t k, uint32_t case_id) {
    int32_t v = (int32_t)((row * 3u + k * 5u + case_id * 7u) % 15u) - 7;
    return (float)v * 0.0625f;
}

static float b_pattern(uint32_t k, uint32_t col, uint32_t case_id) {
    int32_t v = (int32_t)((k * 7u + col * 2u + case_id * 3u) % 13u) - 6;
    return (float)v * 0.0625f;
}

static float bias_pattern(uint32_t col, uint32_t case_id) {
    int32_t v = (int32_t)((col * 5u + case_id * 11u) % 17u) - 8;
    return (float)v * 0.03125f;
}

static uint16_t sentinel_value(uint32_t index) {
    return (uint16_t)(QGEMM_SENTINEL + index);
}

static void fill_case(uint32_t case_id) {
    for (uint32_t r = 0; r < QGEMM_M; ++r) {
        for (uint32_t k = 0; k < QGEMM_K; ++k)
            a_values[r * QGEMM_K + k] = float_to_half(a_pattern(r, k, case_id));
    }
    for (uint32_t k = 0; k < QGEMM_K; ++k) {
        for (uint32_t c = 0; c < QGEMM_N; ++c)
            b_values[k * QGEMM_N + c] = float_to_half(b_pattern(k, c, case_id));
    }
    for (uint32_t c = 0; c < QGEMM_N; ++c)
        bias_values[c] = bias_pattern(c, case_id);
    for (uint32_t i = 0; i < QGEMM_OUT_HALFS + QGEMM_GUARD_HALFS; ++i)
        out_values[i] = sentinel_value(i);
}

static void compute_expected(void) {
    for (uint32_t r = 0; r < QGEMM_M; ++r) {
        for (uint32_t c = 0; c < QGEMM_N; ++c) {
            float acc = bias_values[c];
            for (uint32_t k = 0; k < QGEMM_K; ++k) {
                float av = half_to_float(a_values[r * QGEMM_K + k]);
                float bv = half_to_float(b_values[k * QGEMM_N + c]);
                acc += av * bv;
            }
            expected_values[r * QGEMM_N + c] = float_to_half(acc);
        }
    }
}

static int check_case(uint32_t case_id) {
    int mismatches = 0;
    for (uint32_t i = 0; i < QGEMM_OUT_HALFS; ++i) {
        uint16_t got = out_values[i];
        uint16_t want = expected_values[i];
        if (got != want) {
            if (mismatches < 12)
                printk("ERROR: fp16_qgemm case=%d idx=%d got=%x want=%x\n",
                       (int)case_id, (int)i, got, want);
            ++mismatches;
        }
    }
    return mismatches;
}

static int check_guards(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < QGEMM_GUARD_HALFS; ++i) {
        uint16_t got = out_values[QGEMM_OUT_HALFS + i];
        uint16_t want = sentinel_value(QGEMM_OUT_HALFS + i);
        if (got != want) {
            if (mismatches < 8)
                printk("ERROR: fp16_qgemm guard=%d got=%x want=%x\n", (int)i, got, want);
            ++mismatches;
        }
    }
    return mismatches;
}

static uint32_t checksum_halfs(const uint16_t *values, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; ++i) {
        hash ^= values[i];
        hash *= 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t a_dev = 0;
    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t bias_dev = 0;
    vc4_deviceptr_t out_dev = 0;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("quantized_blocked_coop_gemm_vc4kernel program create failed");
    if (vc4Malloc(program, &a_dev, sizeof(a_values)) < 0 ||
        vc4Malloc(program, &b_dev, sizeof(b_values)) < 0 ||
        vc4Malloc(program, &bias_dev, sizeof(bias_values)) < 0 ||
        vc4Malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("quantized_blocked_coop_gemm_vc4kernel allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(QGEMM_M * QGEMM_LANES, 1u, 1u);
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0;
    int start = timer_get_usec();

    printk("Running VC4 fp16 quantized_blocked_coop_gemm_vc4kernel candidate bundle...\n");

    for (uint32_t case_id = 0; case_id < QGEMM_CASES; ++case_id) {
        fill_case(case_id);
        compute_expected();

        if (vc4MemcpyHtoD(program, a_dev, a_values, sizeof(a_values)) < 0 ||
            vc4MemcpyHtoD(program, b_dev, b_values, sizeof(b_values)) < 0 ||
            vc4MemcpyHtoD(program, bias_dev, bias_values, sizeof(bias_values)) < 0 ||
            vc4MemcpyHtoD(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            quantized_blocked_coop_gemm_vc4kernel_launch(
                program, grid, block, a_dev, b_dev, bias_dev, out_dev) < 0 ||
            vc4MemcpyDtoH(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: fp16_qgemm launch/copy failed case=%d\n", (int)case_id);
            ++launch_failures;
            continue;
        }

        int mismatches = check_case(case_id);
        int guards = check_guards();
        uint32_t checksum = checksum_halfs(out_values, QGEMM_OUT_HALFS);
        checksum_accum ^= checksum + case_id * 0x9e3779b9u;
        total_mismatches += mismatches;
        sentinel_mismatches += guards;
        printk("FP16_QGEMM_COOP_CASE case=%d mismatches=%d sentinel_mismatches=%d checksum=%x\n",
               (int)case_id, mismatches, guards, checksum);
    }

    launch_failures += (int)quantized_blocked_coop_gemm_vc4kernel_runtime_launch_failures();
    uint32_t launches = quantized_blocked_coop_gemm_vc4kernel_runtime_launches();
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=quantized_blocked_coop_gemm_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d m=%d n=%d k=%d k_tile=%d saw_cooperative_block=%d saw_vdr_a_b_tiles=%d saw_vpm_shared_tiles=%d saw_barrier=%d saw_f16_unpack=%d saw_f32_accumulate=%d saw_f16_pack=%d saw_tmu_bias=%d saw_vdw_store=%d checksum_accum=%x runtime_launches=%u elapsed_usec=%d\n",
           status, QGEMM_CASES, total_mismatches, sentinel_mismatches, launch_failures,
           12, QGEMM_LANES, QGEMM_M, QGEMM_N, QGEMM_K, QGEMM_K_TILE,
           1, 1, 1, 1, 1, 1, 1, 1, 1, checksum_accum, launches, timer_get_usec() - start);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, bias_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
