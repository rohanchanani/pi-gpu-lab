#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 12u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GROUPS 11u
#define GROUP_WORDS (GROUPS * LANES)
#define MAX_N 193u
#define MAX_QPUS 24u
#define BUFFER_N (MAX_QPUS * GROUP_WORDS + 64u)
#define INPUT_N 384u
#define SENTINEL 0xc0010000u

struct i32_case {
    uint32_t n;
    uint32_t stride;
    uint32_t seed;
};

static const struct i32_case cases[] = {
    {0u, 0x00010003u, 0xffffffffu},
    {1u, 0x80000001u, 0x0000fffdu},
    {15u, 0xffffffefu, 0x70000003u},
    {16u, 0x00010001u, 0x80000000u},
    {17u, 0x7fffffffu, 0xffffffffu},
    {31u, 0x12345678u, 0x87654321u},
    {32u, 0xfffffffeu, 0xfffffffdu},
    {33u, 0x00ffffffu, 0x00010005u},
    {65u, 0x01000001u, 0x00010003u},
    {193u, 0xdeadbeefu, 0x13579bdfu}
};

static uint32_t input_values[INPUT_N];
static uint32_t out_values[BUFFER_N];

static uint32_t input_value(uint32_t index, uint32_t seed, uint32_t stride) {
    return 0x80000000u ^ (index * 1103515245u + seed + stride * 17u);
}

static void fill_input(uint32_t seed, uint32_t stride) {
    for (uint32_t i = 0; i < INPUT_N; ++i)
        input_values[i] = input_value(i, seed, stride);
}

static uint32_t sentinel_value(uint32_t index) {
    return SENTINEL + index;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = sentinel_value(i);
}

static uint32_t scalar_ctrl(uint32_t seed, uint32_t stride) {
    uint32_t mul = seed * stride;
    uint32_t and_v = mul & 0x00ff00ffu;
    uint32_t or_v = and_v | 0x55000000u;
    uint32_t xor_v = or_v ^ seed;
    uint32_t shl = xor_v << 3u;
    uint32_t shru = xor_v >> 5u;
    uint32_t shrs = (uint32_t)((int32_t)xor_v >> 5);
    uint32_t min_s = ((int32_t)shrs < (int32_t)shl) ? shrs : shl;
    uint32_t max_u = shru > mul ? shru : mul;
    int slt = (int32_t)seed < (int32_t)stride;
    int ult = seed < stride;
    int low = (mul & 1u) != 0u;
    uint32_t bool_i = (uint32_t)((slt ^ ult) | low);
    return slt ? (max_u + bool_i) : (min_s - bool_i);
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static uint32_t active_qpus_for_waves(uint32_t waves) {
    return waves * ACTIVE_QPUS;
}

static uint32_t out_index(uint32_t qpu, uint32_t group, uint32_t lane) {
    return qpu * GROUP_WORDS + group * LANES + lane;
}

static uint32_t reduce_add(const uint32_t *values, uint32_t count) {
    uint32_t r = 0u;
    for (uint32_t i = 0; i < count; ++i)
        r += values[i];
    return r;
}

static uint32_t reduce_min_s(const uint32_t *values, uint32_t count) {
    int32_t r = (int32_t)values[0];
    for (uint32_t i = 1; i < count; ++i)
        if ((int32_t)values[i] < r)
            r = (int32_t)values[i];
    return (uint32_t)r;
}

static uint32_t reduce_max_s(const uint32_t *values, uint32_t count) {
    int32_t r = (int32_t)values[0];
    for (uint32_t i = 1; i < count; ++i)
        if ((int32_t)values[i] > r)
            r = (int32_t)values[i];
    return (uint32_t)r;
}

static uint32_t reduce_min_u(const uint32_t *values, uint32_t count) {
    uint32_t r = values[0];
    for (uint32_t i = 1; i < count; ++i)
        if (values[i] < r)
            r = values[i];
    return r;
}

static uint32_t reduce_max_u(const uint32_t *values, uint32_t count) {
    uint32_t r = values[0];
    for (uint32_t i = 1; i < count; ++i)
        if (values[i] > r)
            r = values[i];
    return r;
}

static uint32_t reduce_and(const uint32_t *values, uint32_t count) {
    uint32_t r = 0xffffffffu;
    for (uint32_t i = 0; i < count; ++i)
        r &= values[i];
    return r;
}

static uint32_t reduce_or(const uint32_t *values, uint32_t count) {
    uint32_t r = 0u;
    for (uint32_t i = 0; i < count; ++i)
        r |= values[i];
    return r;
}

static uint32_t reduce_xor(const uint32_t *values, uint32_t count) {
    uint32_t r = 0u;
    for (uint32_t i = 0; i < count; ++i)
        r ^= values[i];
    return r;
}

static uint32_t reduce_cmp_mask_or(const uint32_t *values, uint32_t count) {
    uint32_t r = 0u;
    for (uint32_t i = 0; i < count; ++i)
        if ((int32_t)values[i] < 0)
            r |= values[i];
    return r;
}

static uint32_t expected_group(uint32_t group, const uint32_t *values,
                               uint32_t count, uint32_t ctrl) {
    switch (group) {
    case 0: return reduce_add(values, count);
    case 1: return reduce_min_s(values, count);
    case 2: return reduce_max_s(values, count);
    case 3: return reduce_min_u(values, count);
    case 4: return reduce_max_u(values, count);
    case 5: return reduce_and(values, count);
    case 6: return reduce_or(values, count);
    case 7: return reduce_xor(values, count);
    case 8: return ctrl;
    case 10: return reduce_cmp_mask_or(values, count);
    default: return 0u;
    }
}

static int verify_qpu(uint32_t qpu, uint32_t n, uint32_t ctrl) {
    uint32_t values[LANES];
    uint32_t base = qpu * LANES;
    uint32_t count = 0u;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t index = base + lane;
        if (index < n)
            values[count++] = input_values[index];
    }

    int mismatches = 0;
    if (count != 0u) {
        for (uint32_t group = 0; group < GROUPS; ++group) {
            if (group == 9u)
                continue;
            uint32_t expected = expected_group(group, values, count, ctrl);
            for (uint32_t lane = 0; lane < count; ++lane) {
                uint32_t idx = out_index(qpu, group, lane);
                if (out_values[idx] != expected) {
                    if (mismatches < 8)
                        printk("ERROR: mixed_i32 qpu=%d group=%d lane=%d gpu=%x expected=%x count=%d\n",
                               (int)qpu, (int)group, (int)lane,
                               out_values[idx], expected, (int)count);
                    mismatches++;
                }
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t active_qpus, uint32_t n) {
    int mismatches = 0;
    for (uint32_t qpu = 0; qpu < MAX_QPUS; ++qpu) {
        uint32_t base = qpu * LANES;
        uint32_t active = 0u;
        if (qpu < active_qpus && base < n) {
            uint32_t remaining = n - base;
            active = remaining > LANES ? LANES : remaining;
        }
        for (uint32_t group = 0; group < GROUPS; ++group) {
            for (uint32_t lane = 0; lane < LANES; ++lane) {
                uint32_t idx = out_index(qpu, group, lane);
                int should_be_written = 0;
                if (qpu < active_qpus && group == 9u)
                    should_be_written = 1;
                else if (qpu < active_qpus && group != 9u && lane < active)
                    should_be_written = 1;
                if (!should_be_written && out_values[idx] != sentinel_value(idx)) {
                    if (mismatches < 8)
                        printk("ERROR: mixed_i32 sentinel qpu=%d group=%d lane=%d gpu=%x expected=%x\n",
                               (int)qpu, (int)group, (int)lane,
                               out_values[idx], sentinel_value(idx));
                    mismatches++;
                }
            }
        }
    }
    for (uint32_t i = MAX_QPUS * GROUP_WORDS; i < BUFFER_N; ++i) {
        if (out_values[i] != sentinel_value(i)) {
            if (mismatches < 8)
                printk("ERROR: mixed_i32 tail sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], sentinel_value(i));
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_empty_identity(uint32_t active_qpus) {
    int mismatches = 0;
    for (uint32_t qpu = 0; qpu < active_qpus; ++qpu) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t idx = out_index(qpu, 9u, lane);
            if (out_values[idx] != 0u) {
                if (mismatches < 8)
                    printk("ERROR: mixed_i32 empty identity qpu=%d lane=%d gpu=%x\n",
                           (int)qpu, (int)lane, out_values[idx]);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t active_qpus) {
    int checksum = 0;
    for (uint32_t qpu = 0; qpu < active_qpus; ++qpu)
        for (uint32_t group = 0; group < GROUPS; ++group)
            checksum += (int)(out_values[out_index(qpu, group, 0)] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("mixed_i32 allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_n0 = 0;
    int saw_overflow = 0;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 mixed_i32_control_reduce_scalar_address_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        const struct i32_case *tc = &cases[case_id];
        uint32_t waves = rounded_waves(tc->n);
        uint32_t active_qpus = active_qpus_for_waves(waves);
        uint32_t ctrl = scalar_ctrl(tc->seed, tc->stride);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_input(tc->seed, tc->stride);
        fill_output();
        if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            mixed_i32_control_reduce_scalar_address_vc4kernel_launch(
                program, grid, block, input_dev, out_dev, tc->n, tc->stride,
                tc->seed) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: mixed_i32 launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)tc->n);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        for (uint32_t qpu = 0; qpu < active_qpus; ++qpu)
            mismatches += verify_qpu(qpu, tc->n, ctrl);
        int empty_mismatches = verify_empty_identity(active_qpus);
        int sentinels = verify_sentinels(active_qpus, tc->n);
        total_mismatches += mismatches + empty_mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum_low16(active_qpus);
        if (tc->n == 0u)
            saw_n0 = 1;
        if ((uint64_t)tc->seed * (uint64_t)tc->stride > 0xffffffffull)
            saw_overflow = 1;
        printk("MIXED_I32_CASE case=%d n=%d waves=%d active_qpus=%d ctrl=%x mismatches=%d empty_mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)tc->n, (int)waves, (int)active_qpus,
               ctrl, mismatches, empty_mismatches, sentinels,
               checksum_low16(active_qpus));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_n0 && saw_overflow)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_i32_control_reduce_scalar_address_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d saw_scalar_mul32=%d saw_scalar_signed_unsigned_cmp=1 saw_i32_reduce_all_kinds=1 saw_empty_predicate_identity=%d saw_vdw_preserve=1 checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           (int)ACTIVE_QPUS, (int)LANES, (int)MAX_N, saw_overflow, saw_n0,
           checksum_accum, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
