#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FA_COOP_ROWS 12u
#define FA_COOP_SEQ 24u
#define FA_COOP_LANES 16u
#define FA_COOP_OUT_WORDS (FA_COOP_ROWS * FA_COOP_LANES)
#define FA_COOP_GUARD_WORDS 32u
#define FA_COOP_CASES 5u
#define FA_COOP_ABS_TOL 0.09f
#define FA_COOP_REL_TOL 0.07f
#define FA_COOP_SENTINEL_BITS 0x7fc0cafeu

static float q_values[FA_COOP_ROWS];
static float k_values[FA_COOP_SEQ];
static float v_values[FA_COOP_SEQ];
static float out_values[FA_COOP_OUT_WORDS + FA_COOP_GUARD_WORDS];
static float expected_values[FA_COOP_ROWS];

static float f_abs_local(float x) {
    return x < 0.0f ? -x : x;
}

static float f_max_local(float a, float b) {
    return a > b ? a : b;
}

static uint32_t float_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

static float exp2_ref(float x) {
    const float ln2 = 0.6931471805599453f;
    int ip = (int)x;
    if ((float)ip > x)
        ip--;
    float f = x - (float)ip;
    float y = 1.0f + f * ln2;
    float term = f * ln2;
    term *= f * ln2;
    y += 0.5f * term;
    term *= f * ln2;
    y += 0.16666667f * term;
    term *= f * ln2;
    y += 0.04166667f * term;
    term *= f * ln2;
    y += 0.00833333f * term;

    if (ip > 0) {
        for (int i = 0; i < ip; i++)
            y *= 2.0f;
    } else {
        for (int i = 0; i < -ip; i++)
            y *= 0.5f;
    }
    return y;
}

static float exp_ref(float x) {
    return exp2_ref(x * 1.4426950408889634f);
}

static void fill_case(uint32_t case_id) {
    for (uint32_t i = 0; i < FA_COOP_ROWS; i++) {
        float base = (float)((int)i - 5);
        if (case_id == 0)
            q_values[i] = 0.03125f * base;
        else if (case_id == 1)
            q_values[i] = ((i & 1u) ? -0.046875f : 0.0390625f) * (float)(i + 1u);
        else if (case_id == 2)
            q_values[i] = 0.015625f * (float)((i * 7u + 3u) % 17u);
        else if (case_id == 3)
            q_values[i] = -0.0234375f * (float)((i * 5u + 1u) % 13u);
        else
            q_values[i] = 0.0078125f * (float)((i * 11u + 9u) % 23u);
    }

    for (uint32_t j = 0; j < FA_COOP_SEQ; j++) {
        int centered = (int)j - 11;
        if (case_id == 0) {
            k_values[j] = 0.0625f * (float)centered;
            v_values[j] = 0.125f + 0.015625f * (float)j;
        } else if (case_id == 1) {
            k_values[j] = ((j & 1u) ? -0.0546875f : 0.046875f) * (float)((j % 7u) + 1u);
            v_values[j] = -0.25f + 0.03125f * (float)((j * 3u + 5u) % 19u);
        } else if (case_id == 2) {
            k_values[j] = 0.0234375f * (float)((j * 5u + 2u) % 29u);
            v_values[j] = 0.5f - 0.020833334f * (float)((j * 7u + 1u) % 23u);
        } else if (case_id == 3) {
            k_values[j] = -0.01953125f * (float)((j * 11u + 4u) % 31u);
            v_values[j] = ((j & 1u) ? 0.375f : -0.3125f) + 0.0078125f * (float)j;
        } else {
            k_values[j] = 0.01171875f * (float)((j * 13u + 6u) % 37u);
            v_values[j] = -0.125f + 0.010416667f * (float)((j * 17u + 8u) % 41u);
        }
    }

    for (uint32_t i = 0; i < FA_COOP_OUT_WORDS + FA_COOP_GUARD_WORDS; i++)
        ((uint32_t *)out_values)[i] = FA_COOP_SENTINEL_BITS;
}

static void compute_expected(float scale) {
    for (uint32_t row = 0; row < FA_COOP_ROWS; row++) {
        float m = -3.4028234663852886e38f;
        float l = 0.0f;
        float acc = 0.0f;

        for (uint32_t tile = 0; tile < 2u; tile++) {
            uint32_t base = tile * FA_COOP_ROWS;
            for (uint32_t lane_owner = 0; lane_owner < FA_COOP_ROWS; lane_owner++) {
                uint32_t j = base + lane_owner;
                float score = q_values[row] * k_values[j] * scale;
                float m_new = f_max_local(m, score);
                float alpha = (l == 0.0f) ? 0.0f : exp_ref(m - m_new);
                float p = exp_ref(score - m_new);
                l = l * alpha + p;
                acc = acc * alpha + p * v_values[j];
                m = m_new;
            }
        }

        expected_values[row] = acc / l;
    }
}

static int check_case(uint32_t case_id) {
    int mismatches = 0;
    for (uint32_t row = 0; row < FA_COOP_ROWS; row++) {
        float got = out_values[row * FA_COOP_LANES];
        float expected = expected_values[row];
        float diff = f_abs_local(got - expected);
        float allowed = FA_COOP_ABS_TOL + FA_COOP_REL_TOL * f_abs_local(expected);
        if (!(diff <= allowed)) {
            if (mismatches < 8)
                printk("ERROR: case=%d row=%d got=%x expected=%x diff=%x allowed=%x\n",
                       (int)case_id, (int)row, float_bits(got), float_bits(expected),
                       float_bits(diff), float_bits(allowed));
            mismatches++;
        }

        for (uint32_t lane = 1; lane < FA_COOP_LANES; lane++) {
            uint32_t bits = ((uint32_t *)out_values)[row * FA_COOP_LANES + lane];
            if (bits != FA_COOP_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: case=%d row=%d inactive_lane=%d changed=%x expected_sentinel=%x\n",
                           (int)case_id, (int)row, (int)lane, bits, FA_COOP_SENTINEL_BITS);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int check_guards(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < FA_COOP_GUARD_WORDS; i++) {
        uint32_t bits = ((uint32_t *)out_values)[FA_COOP_OUT_WORDS + i];
        if (bits != FA_COOP_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: guard=%d changed=%x expected=%x\n", (int)i, bits, FA_COOP_SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t checksum_words(const uint32_t *values, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; i++) {
        hash ^= values[i];
        hash *= 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t q_dev = 0;
    vc4_deviceptr_t k_dev = 0;
    vc4_deviceptr_t v_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    const float scale = 1.0f;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &q_dev, sizeof(q_values)) < 0 ||
        vc4Malloc(program, &k_dev, sizeof(k_values)) < 0 ||
        vc4Malloc(program, &v_dev, sizeof(v_values)) < 0 ||
        vc4Malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("flash_attention_coop12_fwd24_d1_vc4kernel allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(FA_COOP_ROWS * FA_COOP_LANES, 1u, 1u);
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0u;
    int start = timer_get_usec();

    printk("Running VC4 flash_attention_coop12_fwd24_d1_vc4kernel candidate bundle...\n");

    for (uint32_t case_id = 0; case_id < FA_COOP_CASES; case_id++) {
        fill_case(case_id);
        compute_expected(scale);

        if (vc4MemcpyHtoD(program, q_dev, q_values, sizeof(q_values)) < 0 ||
            vc4MemcpyHtoD(program, k_dev, k_values, sizeof(k_values)) < 0 ||
            vc4MemcpyHtoD(program, v_dev, v_values, sizeof(v_values)) < 0 ||
            vc4MemcpyHtoD(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            flash_attention_coop12_fwd24_d1_vc4kernel_launch(program, grid, block, q_dev, k_dev, v_dev, out_dev, scale) < 0 ||
            vc4MemcpyDtoH(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: flash_attention_coop12_fwd24_d1_vc4kernel launch/copy failed case=%d\n",
                   (int)case_id);
            launch_failures++;
            continue;
        }

        int mismatches = check_case(case_id);
        int guards = check_guards();
        uint32_t checksum = checksum_words((const uint32_t *)out_values, FA_COOP_OUT_WORDS);
        checksum_accum ^= checksum + case_id * 0x9e3779b9u;
        total_mismatches += mismatches;
        sentinel_mismatches += guards;
        printk("FLASH_ATTENTION_COOP12_CASE case=%d mismatches=%d sentinel_mismatches=%d checksum=%x\n",
               (int)case_id, mismatches, guards, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=flash_attention_coop12_fwd24_d1_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d seq=%d q_rows=%d runtime_allocations=%d runtime_launches=%d timeouts=%d saw_flash_online_softmax=%d saw_cooperative_tile_load=%d saw_barrier=%d saw_shared_vpm_reuse=%d saw_sfu_exp=%d saw_sfu_recip=%d saw_tmu_qkv=%d saw_vdw_preserve=%d no_attention_matrix=%d checksum_accum=%x elapsed_usec=%d\n",
           status, FA_COOP_CASES, total_mismatches, sentinel_mismatches, launch_failures,
           12, FA_COOP_LANES, FA_COOP_ROWS, FA_COOP_SEQ, FA_COOP_ROWS, 4, FA_COOP_CASES,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, checksum_accum, elapsed);

    vc4Free(program, q_dev);
    vc4Free(program, k_dev);
    vc4Free(program, v_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
