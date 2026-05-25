#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FLASH_ATTENTION_TINY_CASES 6u
#define FLASH_ATTENTION_TINY_MAX_Q_LEN 4u
#define FLASH_ATTENTION_TINY_MAX_K_LEN 8u
#define FLASH_ATTENTION_TINY_MAX_D 16u
#define FLASH_ATTENTION_TINY_MAX_Q_WORDS (FLASH_ATTENTION_TINY_MAX_Q_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_K_WORDS (FLASH_ATTENTION_TINY_MAX_K_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_V_WORDS (FLASH_ATTENTION_TINY_MAX_K_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_OUT_WORDS (FLASH_ATTENTION_TINY_MAX_Q_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_GUARD_WORDS 64u
#define FLASH_ATTENTION_TINY_SENTINEL (-23456.25f)
#define FLASH_ATTENTION_TINY_TOLERANCE 0.08f
#define FLASH_ATTENTION_TINY_ACTIVE_QPUS 12u
#define FLASH_ATTENTION_TINY_LANES 16u

struct flash_attention_tiny_case { uint32_t q_len, k_len, d; float scale; uint32_t equal_scores, increasing_scores; };

static const struct flash_attention_tiny_case flash_cases[FLASH_ATTENTION_TINY_CASES] = {
    {0u, 4u, 8u, 0.353553f, 0u, 0u}, {2u, 0u, 8u, 0.353553f, 0u, 0u},
    {1u, 1u, 1u, 1.0f, 0u, 0u}, {2u, 3u, 4u, 0.5f, 1u, 0u},
    {3u, 4u, 8u, 0.353553f, 0u, 1u}, {4u, 8u, 16u, 0.25f, 0u, 0u},
};

static float q_values[FLASH_ATTENTION_TINY_MAX_Q_WORDS];
static float k_values[FLASH_ATTENTION_TINY_MAX_K_WORDS];
static float v_values[FLASH_ATTENTION_TINY_MAX_V_WORDS];
static float out_values[FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS];
static float expected_values[FLASH_ATTENTION_TINY_MAX_OUT_WORDS];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static int invalid_f32(float value) { return !(value == value) || value > 3.4e38f || value < -3.4e38f; }
static float make_q_value(uint32_t q, uint32_t t, uint32_t case_id) { int raw = (int)((q * 11u + t * 7u + case_id * 5u + 3u) % 17u) - 8; return (float)raw * 0.125f; }
static float make_k_value(uint32_t k, uint32_t t, uint32_t case_id) { int raw = (int)((k * 13u + t * 3u + case_id * 9u + 1u) % 19u) - 9; return (float)raw * 0.125f; }
static float make_v_value(uint32_t k, uint32_t t, uint32_t case_id) { int raw = (int)((k * 5u + t * 11u + case_id * 7u + 4u) % 23u) - 11; return (float)raw * 0.0625f; }

static float exp_approx(float x) {
    if (x < -16.0f) return 0.0f;
    if (x > 8.0f) x = 8.0f;
    float y = x * 0.125f, term = 1.0f, sum = 1.0f;
    for (uint32_t i = 1u; i <= 8u; i++) {
        term *= y / (float)i;
        sum += term;
    }
    float e = sum;
    for (uint32_t i = 0; i < 3u; i++) e *= e;
    return e < 0.0f ? 0.0f : e;
}

static void fill_case(uint32_t case_id, const struct flash_attention_tiny_case *tc) {
    for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_Q_WORDS; i++) q_values[i] = 0.0f;
    for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_K_WORDS; i++) k_values[i] = 0.0f;
    for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_V_WORDS; i++) v_values[i] = 0.0f;
    for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS; i++) expected_values[i] = 0.0f;
    for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS; i++) out_values[i] = FLASH_ATTENTION_TINY_SENTINEL;
    for (uint32_t qi = 0; qi < tc->q_len; qi++)
        for (uint32_t t = 0; t < tc->d; t++)
            q_values[qi * tc->d + t] = tc->equal_scores ? 0.0f : (tc->increasing_scores ? ((t == 0u) ? (0.25f + (float)qi * 0.0625f) : 0.0f) : make_q_value(qi, t, case_id));
    for (uint32_t ki = 0; ki < tc->k_len; ki++)
        for (uint32_t t = 0; t < tc->d; t++) {
            k_values[ki * tc->d + t] = tc->equal_scores ? make_k_value(0u, t, case_id) : (tc->increasing_scores ? ((t == 0u) ? (float)(ki + 1u) * 0.25f : 0.0f) : make_k_value(ki, t, case_id));
            v_values[ki * tc->d + t] = make_v_value(ki, t, case_id);
        }
    if (tc->k_len == 0u)
        return;
    for (uint32_t qi = 0; qi < tc->q_len; qi++) {
        float scores[8];
        float max_score = -3.4e38f;
        float sum_exp = 0.0f;
        for (uint32_t ki = 0; ki < tc->k_len; ki++) {
            float dot = 0.0f;
            for (uint32_t t = 0; t < tc->d; t++)
                dot += q_values[qi * tc->d + t] * k_values[ki * tc->d + t];
            scores[ki] = dot * tc->scale;
            if (scores[ki] > max_score) max_score = scores[ki];
        }
        for (uint32_t ki = 0; ki < tc->k_len; ki++) {
            scores[ki] = exp_approx(scores[ki] - max_score);
            sum_exp += scores[ki];
        }
        for (uint32_t t = 0; t < tc->d; t++) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < tc->k_len; ki++)
                acc += (scores[ki] / sum_exp) * v_values[ki * tc->d + t];
            expected_values[qi * tc->d + t] = acc;
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t q_dev = 0, k_dev = 0, v_dev = 0, out_dev = 0;
    uint32_t q_bytes = FLASH_ATTENTION_TINY_MAX_Q_WORDS * sizeof(float);
    uint32_t k_bytes = FLASH_ATTENTION_TINY_MAX_K_WORDS * sizeof(float);
    uint32_t v_bytes = FLASH_ATTENTION_TINY_MAX_V_WORDS * sizeof(float);
    uint32_t out_bytes = (FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS) * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &q_dev, q_bytes) < 0 || vc4_m2_malloc(program, &k_dev, k_bytes) < 0 ||
        vc4_m2_malloc(program, &v_dev, v_bytes) < 0 || vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("flash_attention_tiny device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(FLASH_ATTENTION_TINY_LANES, 1u, 1u);
    printk("FLASH_ATTENTION_TINY_RUNTIME_SETUP max_q_len=%d max_k_len=%d max_d=%d allocations=1\n",
           FLASH_ATTENTION_TINY_MAX_Q_LEN, FLASH_ATTENTION_TINY_MAX_K_LEN, FLASH_ATTENTION_TINY_MAX_D);

    for (uint32_t case_id = 0; case_id < FLASH_ATTENTION_TINY_CASES; case_id++) {
        const struct flash_attention_tiny_case *tc = &flash_cases[case_id];
        uint32_t logical_outputs = tc->q_len * tc->d;
        uint32_t mismatches = 0, case_sentinel = 0;
        float max_abs_diff = 0.0f;
        int checksum = 0;
        fill_case(case_id, tc);
        vc4_dim3 grid = vc4_m2_dim3(tc->q_len == 0u ? 0u : tc->q_len, 1u, 1u);

        if (vc4_m2_copy_htod(program, q_dev, q_values, q_bytes) < 0 ||
            vc4_m2_copy_htod(program, k_dev, k_values, k_bytes) < 0 ||
            vc4_m2_copy_htod(program, v_dev, v_values, v_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            flash_attention_tiny_launch(program, grid, block, q_dev, k_dev, v_dev, out_dev, tc->q_len, tc->k_len, tc->d, tc->scale) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            launch_failures++;
            printk("FLASH_ATTENTION_TINY_CASE case=%d q_len=%d k_len=%d d=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)tc->q_len, (int)tc->k_len, (int)tc->d, (int)(case_id + 1u));
            continue;
        }

        for (uint32_t i = 0; i < logical_outputs; i++) {
            float diff = out_values[i] - expected_values[i];
            float adiff = abs_f32(diff);
            checksum += (int)(out_values[i] * 1024.0f);
            checked_elements++;
            if (adiff > max_abs_diff) max_abs_diff = adiff;
            if (invalid_f32(out_values[i]) || adiff > FLASH_ATTENTION_TINY_TOLERANCE) {
                if (mismatches < 8u)
                    printk("ERROR: case=%d q=%d lane=%d gpu=%f cpu=%f diff=%f\n",
                           (int)case_id, (int)(tc->d == 0u ? 0u : i / tc->d), (int)(tc->d == 0u ? 0u : i % tc->d),
                           out_values[i], expected_values[i], diff);
                mismatches++;
            }
        }
        for (uint32_t i = logical_outputs; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS; i++)
            if (out_values[i] != FLASH_ATTENTION_TINY_SENTINEL)
                case_sentinel++;
        if (max_abs_diff > global_max_abs_diff) global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;
        printk("FLASH_ATTENTION_TINY_CASE case=%d q_len=%d k_len=%d d=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)tc->q_len, (int)tc->k_len, (int)tc->d, (int)logical_outputs,
               (int)mismatches, (int)case_sentinel, checksum, max_abs_diff, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u &&
                          checked_elements > 0u && global_max_abs_diff <= FLASH_ATTENTION_TINY_TOLERANCE) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=flash_attention_tiny status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_q_len=%d max_k_len=%d max_d=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, FLASH_ATTENTION_TINY_CASES, (int)checked_elements, (int)total_mismatches, (int)sentinel_mismatches,
           (int)launch_failures, FLASH_ATTENTION_TINY_ACTIVE_QPUS, FLASH_ATTENTION_TINY_LANES,
           FLASH_ATTENTION_TINY_MAX_Q_LEN, FLASH_ATTENTION_TINY_MAX_K_LEN, FLASH_ATTENTION_TINY_MAX_D,
           checksum_accum, global_max_abs_diff, 1, FLASH_ATTENTION_TINY_CASES, elapsed);
    vc4Free(program, q_dev); vc4Free(program, k_dev); vc4Free(program, v_dev); vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
