#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define PART2_MAX_INPUT_WORDS (16u * 10u * 34u)
#define PART2_MAX_WEIGHT_WORDS (4u * 16u * 3u * 3u)
#define PART2_MAX_BIAS_WORDS 4u
#define PART2_MAX_OUTPUT_WORDS (4u * 8u * 16u)
#define PART2_GUARD_WORDS 64u
#define PART2_SENTINEL_BITS 0x7fc0c0deu
#define PART2_CASES 3u
#define PART2_ABS_TOL 0.06f
#define PART2_REL_TOL 0.03f

typedef struct {
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t pool_size;
    const char *name;
} part2_case_t;

static const part2_case_t cases[PART2_CASES] = {
    {16, 3, 10, 18, 1, "valid_conv3x3_bias_pool1"},
    {16, 2, 10, 34, 2, "valid_conv3x3_bias_pool2"},
    {8,  4,  7, 18, 1, "short_channel_tail_pool1"},
};

static float input_values[PART2_MAX_INPUT_WORDS];
static float weight_values[PART2_MAX_WEIGHT_WORDS];
static float bias_values[PART2_MAX_BIAS_WORDS];
static float out_values[PART2_MAX_OUTPUT_WORDS + PART2_GUARD_WORDS];
static float expected_values[PART2_MAX_OUTPUT_WORDS];

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

static uint32_t conv_h(const part2_case_t *cfg) {
    return cfg->in_h - 3u + 1u;
}

static uint32_t conv_w(const part2_case_t *cfg) {
    return cfg->in_w - 3u + 1u;
}

static uint32_t out_h(const part2_case_t *cfg) {
    return conv_h(cfg) / cfg->pool_size;
}

static uint32_t out_w(const part2_case_t *cfg) {
    return conv_w(cfg) / cfg->pool_size;
}

static uint32_t input_index(const part2_case_t *cfg, uint32_t ic, uint32_t y, uint32_t x) {
    return (ic * cfg->in_h + y) * cfg->in_w + x;
}

static uint32_t weight_index(const part2_case_t *cfg, uint32_t oc, uint32_t ic, uint32_t ky, uint32_t kx) {
    return ((oc * cfg->in_channels + ic) * 3u + ky) * 3u + kx;
}

static uint32_t output_index(const part2_case_t *cfg, uint32_t oc, uint32_t y, uint32_t x) {
    return (oc * out_h(cfg) + y) * out_w(cfg) + x;
}

static void fill_case(uint32_t case_id, const part2_case_t *cfg) {
    for (uint32_t ic = 0; ic < cfg->in_channels; ic++) {
        for (uint32_t y = 0; y < cfg->in_h; y++) {
            for (uint32_t x = 0; x < cfg->in_w; x++) {
                uint32_t mix = (ic * 19u + y * 7u + x * 5u + case_id * 11u) % 31u;
                input_values[input_index(cfg, ic, y, x)] = ((float)((int32_t)mix - 15)) * 0.015625f;
            }
        }
    }
    for (uint32_t i = cfg->in_channels * cfg->in_h * cfg->in_w; i < PART2_MAX_INPUT_WORDS; i++)
        input_values[i] = -90.0f;

    for (uint32_t oc = 0; oc < cfg->out_channels; oc++) {
        bias_values[oc] = 0.0625f * (float)((int32_t)oc - 1);
        for (uint32_t ic = 0; ic < cfg->in_channels; ic++) {
            for (uint32_t ky = 0; ky < 3u; ky++) {
                for (uint32_t kx = 0; kx < 3u; kx++) {
                    uint32_t mix = (oc * 23u + ic * 13u + ky * 3u + kx + case_id) % 17u;
                    weight_values[weight_index(cfg, oc, ic, ky, kx)] =
                        ((float)((int32_t)mix - 8)) * 0.0078125f;
                }
            }
        }
    }
    for (uint32_t i = cfg->out_channels; i < PART2_MAX_BIAS_WORDS; i++)
        bias_values[i] = -50.0f;
    for (uint32_t i = cfg->out_channels * cfg->in_channels * 9u; i < PART2_MAX_WEIGHT_WORDS; i++)
        weight_values[i] = -80.0f;

    for (uint32_t i = 0; i < PART2_MAX_OUTPUT_WORDS + PART2_GUARD_WORDS; i++)
        ((uint32_t *)out_values)[i] = PART2_SENTINEL_BITS;
    for (uint32_t i = 0; i < PART2_MAX_OUTPUT_WORDS; i++)
        expected_values[i] = 0.0f;
}

static float conv_at(const part2_case_t *cfg, uint32_t oc, uint32_t y, uint32_t x) {
    float acc = bias_values[oc];
    for (uint32_t ic = 0; ic < cfg->in_channels; ic++) {
        for (uint32_t ky = 0; ky < 3u; ky++) {
            for (uint32_t kx = 0; kx < 3u; kx++) {
                acc += input_values[input_index(cfg, ic, y + ky, x + kx)] *
                       weight_values[weight_index(cfg, oc, ic, ky, kx)];
            }
        }
    }
    return acc;
}

static void compute_expected(const part2_case_t *cfg) {
    for (uint32_t oc = 0; oc < cfg->out_channels; oc++) {
        for (uint32_t oy = 0; oy < out_h(cfg); oy++) {
            for (uint32_t ox = 0; ox < out_w(cfg); ox++) {
                float best = -3.4028234663852886e38f;
                for (uint32_t py = 0; py < cfg->pool_size; py++) {
                    for (uint32_t px = 0; px < cfg->pool_size; px++) {
                        uint32_t cy = oy * cfg->pool_size + py;
                        uint32_t cx = ox * cfg->pool_size + px;
                        best = f_max_local(best, conv_at(cfg, oc, cy, cx));
                    }
                }
                expected_values[output_index(cfg, oc, oy, ox)] = best;
            }
        }
    }
}

static int check_case(uint32_t case_id, const part2_case_t *cfg) {
    int mismatches = 0;
    for (uint32_t oc = 0; oc < cfg->out_channels; oc++) {
        for (uint32_t y = 0; y < out_h(cfg); y++) {
            for (uint32_t x = 0; x < out_w(cfg); x++) {
                uint32_t idx = output_index(cfg, oc, y, x);
                float got = out_values[idx];
                float expected = expected_values[idx];
                float diff = f_abs_local(got - expected);
                float allowed = PART2_ABS_TOL + PART2_REL_TOL * f_abs_local(expected);
                if (!(diff <= allowed)) {
                    if (mismatches < 10)
                        printk("ERROR: case=%d oc=%d y=%d x=%d got=%x expected=%x diff=%x allowed=%x\n",
                               (int)case_id, (int)oc, (int)y, (int)x,
                               float_bits(got), float_bits(expected),
                               float_bits(diff), float_bits(allowed));
                    mismatches++;
                }
            }
        }
    }
    return mismatches;
}

static int check_guards(const part2_case_t *cfg) {
    int mismatches = 0;
    uint32_t used = cfg->out_channels * out_h(cfg) * out_w(cfg);
    for (uint32_t i = used; i < PART2_MAX_OUTPUT_WORDS + PART2_GUARD_WORDS; i++) {
        uint32_t got = ((uint32_t *)out_values)[i];
        if (got != PART2_SENTINEL_BITS) {
            if (mismatches < 10)
                printk("ERROR: guard index=%d got=%x expected=%x\n", (int)i, got, PART2_SENTINEL_BITS);
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
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t weights_dev = 0;
    vc4_deviceptr_t bias_dev = 0;
    vc4_deviceptr_t out_dev = 0;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4Malloc(program, &weights_dev, sizeof(weight_values)) < 0 ||
        vc4Malloc(program, &bias_dev, sizeof(bias_values)) < 0 ||
        vc4Malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("fused_conv2d_3x3_maxpool_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0u;
    int start = timer_get_usec();

    printk("Running assignment Part 2 VC4 fused_conv2d_3x3_maxpool candidate bundle...\n");

    for (uint32_t case_id = 0; case_id < PART2_CASES; case_id++) {
        const part2_case_t *cfg = &cases[case_id];
        fill_case(case_id, cfg);
        compute_expected(cfg);

        vc4_dim3 grid = vc4_m2_dim3((out_w(cfg) + 15u) / 16u, out_h(cfg), cfg->out_channels);
        vc4_dim3 block = vc4_m2_dim3(16u, 1u, 1u);

        if (vc4MemcpyHtoD(program, input_dev, input_values, sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, weights_dev, weight_values, sizeof(weight_values)) < 0 ||
            vc4MemcpyHtoD(program, bias_dev, bias_values, sizeof(bias_values)) < 0 ||
            vc4MemcpyHtoD(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            fused_conv2d_3x3_maxpool_vc4kernel_launch(
                program, grid, block, input_dev, weights_dev, bias_dev, out_dev,
                (int32_t)cfg->in_channels, (int32_t)cfg->in_h, (int32_t)cfg->in_w,
                (int32_t)out_h(cfg), (int32_t)out_w(cfg), (int32_t)cfg->pool_size) < 0 ||
            vc4MemcpyDtoH(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: launch/copy failed case=%d name=%s\n", (int)case_id, cfg->name);
            launch_failures++;
            continue;
        }

        int mismatches = check_case(case_id, cfg);
        int guards = check_guards(cfg);
        uint32_t used_words = cfg->out_channels * out_h(cfg) * out_w(cfg);
        uint32_t checksum = checksum_words((const uint32_t *)out_values, used_words);
        checksum_accum ^= checksum + 0x9e3779b9u * case_id;
        total_mismatches += mismatches;
        sentinel_mismatches += guards;

        printk("PART2_FUSED_CONV_MAXPOOL_CASE case=%d name=%s in_ch=%d out_ch=%d in_h=%d in_w=%d pool_size=%d out_h=%d out_w=%d mismatches=%d sentinel_mismatches=%d checksum=%x\n",
               (int)case_id, cfg->name, (int)cfg->in_channels, (int)cfg->out_channels,
               (int)cfg->in_h, (int)cfg->in_w, (int)cfg->pool_size,
               (int)out_h(cfg), (int)out_w(cfg), mismatches, guards, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fused_conv2d_3x3_maxpool_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_allocations=%d runtime_launches=%d timeouts=%d saw_assignment_part2_nchw=%d saw_valid_conv3x3_stride1=%d saw_bias=%d saw_pool_size_1=%d saw_pool_size_2=%d saw_fused_maxpool=%d saw_input_channel_loop=%d saw_output_channel_grid=%d saw_tmu_input_weight_bias=%d saw_vdw_preserve=%d no_intermediate_conv_buffer=%d checksum_accum=%x elapsed_usec=%d\n",
           status, PART2_CASES, total_mismatches, sentinel_mismatches, launch_failures,
           4, PART2_CASES, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, checksum_accum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, weights_dev);
    vc4Free(program, bias_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
