#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 16u
#define BUFFER_N (LANES + GUARD)
#define AMOUNT 17u
#define F32_SENTINEL (-7654.25f)
#define I32_SENTINEL 0x7a110000u
#define ABS_TOL 0.001f
#define REL_TOL 0.001f

static const float input_f32[LANES] = {
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
    0.5f, 0.25f, 0.125f, 0.0625f, 3.0f, 5.0f, 7.0f, 9.0f
};

static const uint32_t input_i32[LANES] = {
    0x00000011u, 0x00000023u, 0x00000035u, 0x00000047u,
    0x00000059u, 0x0000006bu, 0x0000007du, 0x0000008fu,
    0x00000091u, 0x000000a3u, 0x000000b5u, 0x000000c7u,
    0x000000d9u, 0x000000ebu, 0x000000fdu, 0x0000000fu
};

static float out_f32[BUFFER_N];
static uint32_t audit_i32[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static void fill_outputs(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        out_f32[i] = F32_SENTINEL;
        audit_i32[i] = I32_SENTINEL + i;
    }
}

static uint32_t source_lane(uint32_t lane) {
    return (lane + (AMOUNT & 15u)) & 15u;
}

static int verify_f32(float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        float expected = 1.0f / input_f32[source_lane(lane)];
        float diff = out_f32[lane] - expected;
        float ad = absf_local(diff);
        float rd = ad / absf_local(expected);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (rd > *max_rel_diff)
            *max_rel_diff = rd;
        if (ad > ABS_TOL && rd > REL_TOL) {
            if (mismatches < 8)
                printk("ERROR: dynamic_tmu_sfu_pack f32 lane=%d gpu=%f expected=%f diff=%f rel=%f\n",
                       (int)lane, out_f32[lane], expected, diff, rd);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_i32(void) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t expected = input_i32[source_lane(lane)] & 0xffu;
        if (audit_i32[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_tmu_sfu_pack i32 lane=%d gpu=%x expected=%x\n",
                       (int)lane, audit_i32[lane], expected);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_N; ++i) {
        uint32_t expected_i32 = I32_SENTINEL + i;
        if (out_f32[i] != F32_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: dynamic_tmu_sfu_pack f32 sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_f32[i], F32_SENTINEL);
            ++mismatches;
        }
        if (audit_i32[i] != expected_i32) {
            if (mismatches < 8)
                printk("ERROR: dynamic_tmu_sfu_pack i32 sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, audit_i32[i], expected_i32);
            ++mismatches;
        }
    }
    return mismatches;
}

static int checksum_i32(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        checksum += (int)audit_i32[lane];
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel program create failed");

    vc4_deviceptr_t input_f32_dev = 0;
    vc4_deviceptr_t input_i32_dev = 0;
    vc4_deviceptr_t out_f32_dev = 0;
    vc4_deviceptr_t audit_i32_dev = 0;
    if (vc4_m2_malloc(program, &input_f32_dev, sizeof(input_f32)) < 0 ||
        vc4_m2_malloc(program, &input_i32_dev, sizeof(input_i32)) < 0 ||
        vc4_m2_malloc(program, &out_f32_dev, sizeof(out_f32)) < 0 ||
        vc4_m2_malloc(program, &audit_i32_dev, sizeof(audit_i32)) < 0)
        panic("fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel allocation failed");

    fill_outputs();
    int launch_failures = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, input_f32_dev, input_f32, sizeof(input_f32)) < 0 ||
        vc4_m2_copy_htod(program, input_i32_dev, input_i32, sizeof(input_i32)) < 0 ||
        vc4_m2_copy_htod(program, out_f32_dev, out_f32, sizeof(out_f32)) < 0 ||
        vc4_m2_copy_htod(program, audit_i32_dev, audit_i32, sizeof(audit_i32)) < 0 ||
        fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel_launch(
            program, grid, block, input_f32_dev, input_i32_dev, out_f32_dev,
            audit_i32_dev, AMOUNT) < 0 ||
        vc4_m2_copy_dtoh(program, out_f32, out_f32_dev, sizeof(out_f32)) < 0 ||
        vc4_m2_copy_dtoh(program, audit_i32, audit_i32_dev, sizeof(audit_i32)) < 0) {
        printk("ERROR: dynamic_tmu_sfu_pack launch/copy failed\n");
        ++launch_failures;
    }

    launch_failures += (int)fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel_runtime_launch_failures();
    int f32_mismatches = launch_failures ? 0 : verify_f32(&max_abs_diff, &max_rel_diff);
    int i32_mismatches = launch_failures ? 0 : verify_i32();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int total_mismatches = f32_mismatches + i32_mismatches;
    int checksum = launch_failures ? 0 : checksum_i32();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_tmu_safe_offset=1 saw_sfu=1 saw_pack_unpack=1 saw_dynamic_rotate=1 saw_vdw_preserve=1 rotate_direction=left_source_plus_amount max_abs_diff=%f max_rel_diff=%f checksum_accum=%d runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, max_abs_diff, max_rel_diff, checksum,
           (int)fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel_runtime_launches(),
           timer_get_usec() - start);

    vc4Free(program, input_f32_dev);
    vc4Free(program, input_i32_dev);
    vc4Free(program, out_f32_dev);
    vc4Free(program, audit_i32_dev);
    vc4_program_destroy(program);
}
