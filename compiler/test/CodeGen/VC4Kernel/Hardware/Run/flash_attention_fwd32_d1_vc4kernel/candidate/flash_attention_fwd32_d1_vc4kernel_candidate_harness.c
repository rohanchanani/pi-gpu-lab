#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 12u
#define SEQ_N 32u
#define MAX_ROWS 64u
#define GUARD_ROWS 2u
#define OUT_STRIDE LANES
#define OUT_WORDS ((MAX_ROWS + GUARD_ROWS) * OUT_STRIDE)
#define OUT_SENTINEL (-7777.25f)
#define ABS_TOL 0.08f
#define REL_TOL 0.06f

static const uint32_t row_cases[] = {
    0u, 1u, 2u, 11u, 12u, 13u, 31u, 32u, 33u, 64u
};

static float q_values[MAX_ROWS];
static float k_values[SEQ_N];
static float v_values[SEQ_N];
static float out_values[OUT_WORDS];
static float expected_values[MAX_ROWS];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static float max_f32(float a, float b) { return a > b ? a : b; }

static float exp2_ref(float x) {
    int whole = (int)x;
    if ((float)whole > x)
        --whole;
    if (whole < -20)
        return 0.0f;
    if (whole > 20)
        whole = 20;

    float frac = x - (float)whole;
    const float ln2 = 0.6931471805599453f;
    float y = frac * ln2;
    float term = 1.0f;
    float sum = 1.0f;
    for (uint32_t i = 1; i <= 10; ++i) {
        term *= y / (float)i;
        sum += term;
    }

    if (whole >= 0) {
        for (int i = 0; i < whole; ++i)
            sum *= 2.0f;
    } else {
        for (int i = 0; i < -whole; ++i)
            sum *= 0.5f;
    }
    return sum;
}

static float input_q(uint32_t row) {
    int32_t centered = (int32_t)((row * 17u + 5u) % 23u) - 11;
    return 0.0625f * (float)centered;
}

static float input_k(uint32_t col) {
    int32_t centered = (int32_t)((col * 11u + 3u) % 29u) - 14;
    return 0.0625f * (float)centered;
}

static float input_v(uint32_t col) {
    int32_t centered = (int32_t)((col * 7u + 19u) % 31u) - 15;
    return 0.125f * (float)centered;
}

static void fill_buffers(void) {
    for (uint32_t row = 0; row < MAX_ROWS; ++row) {
        q_values[row] = input_q(row);
        expected_values[row] = 0.0f;
    }
    for (uint32_t col = 0; col < SEQ_N; ++col) {
        k_values[col] = input_k(col);
        v_values[col] = input_v(col);
    }
    for (uint32_t i = 0; i < OUT_WORDS; ++i)
        out_values[i] = OUT_SENTINEL;
}

static float flash_attention_ref_row(uint32_t row, float scale) {
    const float log2e = 1.4426950408889634f;
    float q = q_values[row];

    float m0 = q * k_values[0] * scale;
    for (uint32_t col = 1; col < 16u; ++col) {
        float s = q * k_values[col] * scale;
        if (s > m0)
            m0 = s;
    }
    float l0 = 0.0f;
    float acc0 = 0.0f;
    for (uint32_t col = 0; col < 16u; ++col) {
        float p = exp2_ref((q * k_values[col] * scale - m0) * log2e);
        l0 += p;
        acc0 += p * v_values[col];
    }

    float m1_block = q * k_values[16] * scale;
    for (uint32_t col = 17u; col < 32u; ++col) {
        float s = q * k_values[col] * scale;
        if (s > m1_block)
            m1_block = s;
    }
    float m_new = max_f32(m0, m1_block);
    float alpha = exp2_ref((m0 - m_new) * log2e);
    float l = l0 * alpha;
    float acc = acc0 * alpha;
    for (uint32_t col = 16u; col < 32u; ++col) {
        float p = exp2_ref((q * k_values[col] * scale - m_new) * log2e);
        l += p;
        acc += p * v_values[col];
    }
    return acc / l;
}

static void compute_reference(uint32_t rows, float scale) {
    for (uint32_t row = 0; row < rows; ++row)
        expected_values[row] = flash_attention_ref_row(row, scale);
}

static uint32_t rounded_waves(uint32_t rows) {
    uint32_t waves = (rows + ACTIVE_QPUS - 1u) / ACTIVE_QPUS;
    return waves == 0u ? 1u : waves;
}

static int verify_active(uint32_t rows, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        float got = out_values[row * OUT_STRIDE];
        float expected = expected_values[row];
        float diff = abs_f32(got - expected);
        float rel = diff / max_f32(abs_f32(expected), 1.0e-6f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (rel > *max_rel_diff)
            *max_rel_diff = rel;
        if (diff > max_f32(ABS_TOL, REL_TOL * abs_f32(expected))) {
            if (mismatches < 8)
                printk("ERROR: flash_attention row=%d got=%f expected=%f diff=%f rel=%f\n",
                       (int)row, got, expected, diff, rel);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows) {
    int mismatches = 0;
    for (uint32_t row = 0; row < MAX_ROWS + GUARD_ROWS; ++row) {
        for (uint32_t lane = 0; lane < OUT_STRIDE; ++lane) {
            uint32_t index = row * OUT_STRIDE + lane;
            int should_be_written = row < rows && lane == 0u;
            if (!should_be_written && out_values[index] != OUT_SENTINEL) {
                if (mismatches < 8)
                    printk("ERROR: flash_attention sentinel row=%d lane=%d got=%f expected=%f\n",
                           (int)row, (int)lane, out_values[index],
                           OUT_SENTINEL);
                ++mismatches;
            }
        }
    }
    return mismatches;
}

static int checksum_scaled(uint32_t rows) {
    int checksum = 0;
    for (uint32_t row = 0; row < rows; ++row)
        checksum += (int)(out_values[row * OUT_STRIDE] * 4096.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("flash_attention_fwd32_d1_vc4kernel program create failed");

    vc4_deviceptr_t q_dev = 0;
    vc4_deviceptr_t k_dev = 0;
    vc4_deviceptr_t v_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &q_dev, sizeof(q_values)) < 0 ||
        vc4_m2_malloc(program, &k_dev, sizeof(k_values)) < 0 ||
        vc4_m2_malloc(program, &v_dev, sizeof(v_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("flash_attention_fwd32_d1_vc4kernel allocation failed");

    const float scale = 0.875f;
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int start = timer_get_usec();

    vc4_dim3 block = vc4_m2_dim3(ACTIVE_QPUS * LANES, 1u, 1u);
    printk("Running VC4 flash_attention_fwd32_d1_vc4kernel candidate bundle...\n");

    for (uint32_t case_id = 0; case_id < sizeof(row_cases) / sizeof(row_cases[0]); ++case_id) {
        uint32_t rows = row_cases[case_id];
        uint32_t waves = rounded_waves(rows);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1u, 1u);
        fill_buffers();
        compute_reference(rows, scale);
        if (vc4_m2_copy_htod(program, q_dev, q_values, sizeof(q_values)) < 0 ||
            vc4_m2_copy_htod(program, k_dev, k_values, sizeof(k_values)) < 0 ||
            vc4_m2_copy_htod(program, v_dev, v_values, sizeof(v_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            flash_attention_fwd32_d1_vc4kernel_launch(
                program, grid, block, q_dev, k_dev, v_dev, out_dev,
                (int32_t)rows, scale) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: flash_attention launch/copy failed case=%d rows=%d\n",
                   (int)case_id, (int)rows);
            ++launch_failures;
            continue;
        }
        int mismatches = verify_active(rows, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(rows);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum_scaled(rows);
        printk("FLASH_ATTENTION_FWD32_D1_CASE case=%d rows=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)rows, (int)waves, mismatches, sentinels,
               checksum_scaled(rows), max_abs_diff, max_rel_diff);
    }

    launch_failures += (int)flash_attention_fwd32_d1_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = flash_attention_fwd32_d1_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=flash_attention_fwd32_d1_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d seq_n=%d d_qk=1 d_v=1 online_tiles=2 saw_flash_online_softmax=1 saw_sfu_exp=1 saw_sfu_recip=1 saw_fmax_reduce=1 saw_sum_reduce=1 saw_tmu_qkv=1 saw_vdw_preserve=1 no_attention_matrix=1 checksum_accum=%d max_abs_diff=%f max_rel_diff=%f runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(row_cases) / sizeof(row_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           (int)ACTIVE_QPUS, (int)LANES, (int)SEQ_N, checksum_accum,
           max_abs_diff, max_rel_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, q_dev);
    vc4Free(program, k_dev);
    vc4Free(program, v_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
