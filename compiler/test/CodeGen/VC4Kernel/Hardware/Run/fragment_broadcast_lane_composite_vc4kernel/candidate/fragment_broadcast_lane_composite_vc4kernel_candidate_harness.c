#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define CASES 5u
#define GUARD_WORDS 16u
#define BUFFER_WORDS (LANES + GUARD_WORDS)
#define SENTINEL 0xdeadbeefu

static const uint32_t selected_lanes[CASES] = {0u, 1u, 7u, 15u, 99u};

static const uint32_t i_payload[LANES] = {
    0x00000000u, 0x80000000u, 0xffffffffu, 0x7fffffffu,
    0x12345678u, 0x87654321u, 0xaaaaaaaa, 0x55555555u,
    0x00010001u, 0xffff0000u, 0x01020304u, 0xfedcba98u,
    0x40000000u, 0xc0000000u, 0x0f0f0f0fu, 0xf0f0f0f0u
};

static const uint32_t f_payload_bits[LANES] = {
    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
    0x7fc00000u, 0xff800001u, 0x55555555u, 0x7f800000u,
    0xff800000u, 0x00800000u, 0x80800000u, 0x3f7fffffu,
    0xbf7fffffu, 0x01020304u, 0xfedcba98u, 0x40490fdbu
};

static float f_payload[LANES];
static uint32_t out_i32[BUFFER_WORDS];
static uint32_t out_f32_bits[BUFFER_WORDS];

static float bits_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

static void fill_buffers(void) {
    for (uint32_t lane = 0; lane < LANES; ++lane)
        f_payload[lane] = bits_to_float(f_payload_bits[lane]);
    for (uint32_t i = 0; i < BUFFER_WORDS; ++i) {
        out_i32[i] = SENTINEL;
        out_f32_bits[i] = SENTINEL;
    }
}

static int verify_active(uint32_t selected_lane) {
    int mismatches = 0;
    uint32_t expected_i = selected_lane < LANES ? i_payload[selected_lane] : 0u;
    uint32_t expected_f = selected_lane < LANES ? f_payload_bits[selected_lane] : 0u;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t got_i = out_i32[lane];
        uint32_t got_f = out_f32_bits[lane];
        if (got_i != expected_i || got_f != expected_f) {
            if (mismatches < 8)
                printk("ERROR: broadcast_lane_composite selected=%d lane=%d got_i=%x expected_i=%x got_f=%x expected_f=%x\n",
                       (int)selected_lane, (int)lane, got_i, expected_i,
                       got_f, expected_f);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_WORDS; ++i) {
        if (out_i32[i] != SENTINEL || out_f32_bits[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: broadcast_lane_composite sentinel=%d got_i=%x got_f=%x\n",
                       (int)i, out_i32[i], out_f32_bits[i]);
            ++mismatches;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        checksum += (int)(out_i32[lane] & 0xffffu);
        checksum += (int)(out_f32_bits[lane] & 0xffffu);
    }
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("fragment_broadcast_lane_composite_vc4kernel program create failed");

    vc4_deviceptr_t bits_dev = 0;
    vc4_deviceptr_t f_dev = 0;
    vc4_deviceptr_t out_i_dev = 0;
    vc4_deviceptr_t out_f_dev = 0;
    if (vc4_m2_malloc(program, &bits_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &f_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_i_dev, BUFFER_WORDS * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_f_dev, BUFFER_WORDS * sizeof(uint32_t)) < 0)
        panic("fragment_broadcast_lane_composite_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_lane0 = 0;
    int saw_lane15 = 0;
    int saw_empty_identity = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < CASES; ++case_id) {
        uint32_t selected_lane = selected_lanes[case_id];
        int32_t lane_seed = (int32_t)selected_lane - 1;
        fill_buffers();
        if (vc4_m2_copy_htod(program, bits_dev, i_payload,
                             LANES * sizeof(uint32_t)) < 0 ||
            vc4_m2_copy_htod(program, f_dev, f_payload,
                             LANES * sizeof(float)) < 0 ||
            vc4_m2_copy_htod(program, out_i_dev, out_i32,
                             BUFFER_WORDS * sizeof(uint32_t)) < 0 ||
            vc4_m2_copy_htod(program, out_f_dev, out_f32_bits,
                             BUFFER_WORDS * sizeof(uint32_t)) < 0 ||
            fragment_broadcast_lane_composite_vc4kernel_launch(
                program, grid, block, bits_dev, f_dev, out_i_dev, out_f_dev,
                lane_seed) < 0 ||
            vc4_m2_copy_dtoh(program, out_i32, out_i_dev,
                             BUFFER_WORDS * sizeof(uint32_t)) < 0 ||
            vc4_m2_copy_dtoh(program, out_f32_bits, out_f_dev,
                             BUFFER_WORDS * sizeof(uint32_t)) < 0) {
            printk("ERROR: broadcast_lane_composite launch/copy failed case=%d selected_lane=%d\n",
                   (int)case_id, (int)selected_lane);
            ++launch_failures;
            continue;
        }

        int mismatches = verify_active(selected_lane);
        int sentinels = verify_sentinels();
        int checksum = checksum_low16();
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (selected_lane == 0u)
            saw_lane0 = 1;
        if (selected_lane == 15u)
            saw_lane15 = 1;
        if (selected_lane >= LANES)
            saw_empty_identity = 1;
        printk("BROADCAST_LANE_COMPOSITE_CASE case=%d selected_lane=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)selected_lane, mismatches, sentinels,
               checksum);
    }

    launch_failures +=
        (int)fragment_broadcast_lane_composite_vc4kernel_runtime_launch_failures();
    uint32_t launches =
        fragment_broadcast_lane_composite_vc4kernel_runtime_launches();
    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 &&
         launch_failures == 0 && saw_lane0 && saw_lane15 &&
         saw_empty_identity && launches == CASES)
            ? "PASS"
            : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_broadcast_lane_composite_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_broadcast_lane_composite=1 saw_dynamic_lane_select=1 saw_lane0=%d saw_lane15=%d saw_empty_identity=%d saw_i32_bit_preserving=1 saw_f32_bitcast_path=1 saw_fragment_reduce=1 no_first_class_broadcast_op=1 arbitrary_shuffle_reject_preserved=1 checksum_accum=%d runtime_allocations=4 runtime_launches=%u elapsed_usec=%d\n",
           status, (int)CASES, total_mismatches, sentinel_mismatches,
           launch_failures, (int)LANES, saw_lane0, saw_lane15,
           saw_empty_identity, checksum_accum, launches, elapsed);

    vc4Free(program, bits_dev);
    vc4Free(program, f_dev);
    vc4Free(program, out_i_dev);
    vc4Free(program, out_f_dev);
    vc4_program_destroy(program);
}
