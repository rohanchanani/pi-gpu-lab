#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ATTENTION_QK_NAIVE_CASES 7u
#define ATTENTION_QK_NAIVE_MAX_Q_LEN 7u
#define ATTENTION_QK_NAIVE_MAX_K_LEN 16u
#define ATTENTION_QK_NAIVE_MAX_D 16u
#define ATTENTION_QK_NAIVE_MAX_Q_WORDS (ATTENTION_QK_NAIVE_MAX_Q_LEN * ATTENTION_QK_NAIVE_MAX_D)
#define ATTENTION_QK_NAIVE_MAX_K_WORDS (ATTENTION_QK_NAIVE_MAX_K_LEN * ATTENTION_QK_NAIVE_MAX_D)
#define ATTENTION_QK_NAIVE_MAX_SCORE_WORDS (ATTENTION_QK_NAIVE_MAX_Q_LEN * ATTENTION_QK_NAIVE_MAX_K_LEN)
#define ATTENTION_QK_NAIVE_GUARD_WORDS 64u
#define ATTENTION_QK_NAIVE_SENTINEL (-12345.25f)
#define ATTENTION_QK_NAIVE_EPSILON 0.001f
#define ATTENTION_QK_NAIVE_ACTIVE_QPUS 12u
#define ATTENTION_QK_NAIVE_LANES 16u

struct attention_qk_naive_case { uint32_t q_len, k_len, d; float scale; uint32_t equal_scores; };

static const struct attention_qk_naive_case attention_qk_cases[ATTENTION_QK_NAIVE_CASES] = {
    {0u, 4u, 8u, 0.353553f, 0u}, {2u, 0u, 8u, 0.353553f, 0u},
    {1u, 1u, 1u, 1.0f, 0u}, {2u, 3u, 4u, 0.5f, 0u},
    {3u, 4u, 8u, 0.353553f, 1u}, {5u, 7u, 16u, 0.25f, 0u},
    {7u, 16u, 16u, 0.25f, 0u},
};

static float q_values[ATTENTION_QK_NAIVE_MAX_Q_WORDS];
static float k_values[ATTENTION_QK_NAIVE_MAX_K_WORDS];
static float score_values[ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS];
static float expected_values[ATTENTION_QK_NAIVE_MAX_SCORE_WORDS];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static int invalid_f32(float value) { return !(value == value) || value > 3.4e38f || value < -3.4e38f; }
static float make_q_value(uint32_t q, uint32_t t, uint32_t case_id) { int raw = (int)((q * 11u + t * 7u + case_id * 5u + 3u) % 17u) - 8; return (float)raw * 0.125f; }
static float make_k_value(uint32_t k, uint32_t t, uint32_t case_id) { int raw = (int)((k * 13u + t * 3u + case_id * 9u + 1u) % 19u) - 9; return (float)raw * 0.125f; }

static void fill_case(uint32_t case_id, const struct attention_qk_naive_case *tc) {
    for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_Q_WORDS; i++) q_values[i] = 0.0f;
    for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_K_WORDS; i++) k_values[i] = 0.0f;
    for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS; i++) expected_values[i] = 0.0f;
    for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS; i++) score_values[i] = ATTENTION_QK_NAIVE_SENTINEL;
    for (uint32_t qi = 0; qi < tc->q_len; qi++)
        for (uint32_t t = 0; t < tc->d; t++)
            q_values[qi * tc->d + t] = tc->equal_scores ? 0.0f : make_q_value(qi, t, case_id);
    for (uint32_t ki = 0; ki < tc->k_len; ki++)
        for (uint32_t t = 0; t < tc->d; t++)
            k_values[ki * tc->d + t] = tc->equal_scores ? make_k_value(0u, t, case_id) : make_k_value(ki, t, case_id);
    for (uint32_t qi = 0; qi < tc->q_len; qi++) {
        for (uint32_t ki = 0; ki < tc->k_len; ki++) {
            float dot = 0.0f;
            for (uint32_t t = 0; t < tc->d; t++)
                dot += q_values[qi * tc->d + t] * k_values[ki * tc->d + t];
            expected_values[qi * tc->k_len + ki] = dot * tc->scale;
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t q_dev = 0, k_dev = 0, score_dev = 0;
    uint32_t q_bytes = ATTENTION_QK_NAIVE_MAX_Q_WORDS * sizeof(float);
    uint32_t k_bytes = ATTENTION_QK_NAIVE_MAX_K_WORDS * sizeof(float);
    uint32_t score_bytes = (ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS) * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &q_dev, q_bytes) < 0 || vc4_m2_malloc(program, &k_dev, k_bytes) < 0 ||
        vc4_m2_malloc(program, &score_dev, score_bytes) < 0)
        panic("attention_qk_naive device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(ATTENTION_QK_NAIVE_LANES, 1u, 1u);
    printk("ATTENTION_QK_NAIVE_RUNTIME_SETUP max_q_len=%d max_k_len=%d max_d=%d allocations=1\n",
           ATTENTION_QK_NAIVE_MAX_Q_LEN, ATTENTION_QK_NAIVE_MAX_K_LEN, ATTENTION_QK_NAIVE_MAX_D);

    for (uint32_t case_id = 0; case_id < ATTENTION_QK_NAIVE_CASES; case_id++) {
        const struct attention_qk_naive_case *tc = &attention_qk_cases[case_id];
        uint32_t logical_scores = tc->q_len * tc->k_len;
        uint32_t mismatches = 0, case_sentinel = 0;
        float max_abs_diff = 0.0f;
        int checksum = 0;
        fill_case(case_id, tc);
        vc4_dim3 grid = vc4_m2_dim3((tc->q_len == 0u || tc->k_len == 0u) ? 0u : tc->q_len, 1u, 1u);

        if (vc4_m2_copy_htod(program, q_dev, q_values, q_bytes) < 0 ||
            vc4_m2_copy_htod(program, k_dev, k_values, k_bytes) < 0 ||
            vc4_m2_copy_htod(program, score_dev, score_values, score_bytes) < 0 ||
            attention_qk_naive_launch(program, grid, block, q_dev, k_dev, score_dev, tc->q_len, tc->k_len, tc->d, tc->scale) < 0 ||
            vc4_m2_copy_dtoh(program, score_values, score_dev, score_bytes) < 0) {
            launch_failures++;
            printk("ATTENTION_QK_NAIVE_CASE case=%d q_len=%d k_len=%d d=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)tc->q_len, (int)tc->k_len, (int)tc->d, (int)(case_id + 1u));
            continue;
        }

        for (uint32_t i = 0; i < logical_scores; i++) {
            float diff = score_values[i] - expected_values[i];
            float adiff = abs_f32(diff);
            checksum += (int)(score_values[i] * 1024.0f);
            checked_elements++;
            if (adiff > max_abs_diff) max_abs_diff = adiff;
            if (invalid_f32(score_values[i]) || adiff > ATTENTION_QK_NAIVE_EPSILON) {
                if (mismatches < 8u)
                    printk("ERROR: case=%d q=%d k=%d gpu=%f cpu=%f diff=%f\n",
                           (int)case_id, (int)(tc->k_len == 0u ? 0u : i / tc->k_len), (int)(tc->k_len == 0u ? 0u : i % tc->k_len),
                           score_values[i], expected_values[i], diff);
                mismatches++;
            }
        }
        for (uint32_t i = logical_scores; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS; i++)
            if (score_values[i] != ATTENTION_QK_NAIVE_SENTINEL)
                case_sentinel++;
        if (max_abs_diff > global_max_abs_diff) global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;
        printk("ATTENTION_QK_NAIVE_CASE case=%d q_len=%d k_len=%d d=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)tc->q_len, (int)tc->k_len, (int)tc->d, (int)logical_scores,
               (int)mismatches, (int)case_sentinel, checksum, max_abs_diff, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u &&
                          checked_elements > 0u && global_max_abs_diff <= ATTENTION_QK_NAIVE_EPSILON) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=attention_qk_naive status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_q_len=%d max_k_len=%d max_d=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, ATTENTION_QK_NAIVE_CASES, (int)checked_elements, (int)total_mismatches, (int)sentinel_mismatches,
           (int)launch_failures, ATTENTION_QK_NAIVE_ACTIVE_QPUS, ATTENTION_QK_NAIVE_LANES,
           ATTENTION_QK_NAIVE_MAX_Q_LEN, ATTENTION_QK_NAIVE_MAX_K_LEN, ATTENTION_QK_NAIVE_MAX_D,
           checksum_accum, global_max_abs_diff, 1, ATTENTION_QK_NAIVE_CASES, elapsed);
    vc4Free(program, q_dev);
    vc4Free(program, k_dev);
    vc4Free(program, score_dev);
    vc4_program_destroy(program);
}
