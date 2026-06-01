#include "vc4_m2_candidate_test_helpers.h"
#include <stddef.h>

#define GPU_MEM_FLG 0xC
#define GPU_BASE 0x40000000

#define ERRSTAT_RELEVANT_MASK 0x0000efffu
#define V3D_BASE 0x20C00000
#define V3D_IDENT1 (V3D_BASE + 0x00004)
#define V3D_SQRSV0 (V3D_BASE + 0x00410)
#define V3D_SQRSV1 (V3D_BASE + 0x00414)
#define V3D_DBCFG (V3D_BASE + 0x00e00)
#define V3D_DBQITE (V3D_BASE + 0x00e2c)
#define V3D_DBQITC (V3D_BASE + 0x00e30)
#define V3D_L2CACTL (V3D_BASE + 0x00020)
#define V3D_SLCACTL (V3D_BASE + 0x00024)
#define V3D_VPMBASE (V3D_BASE + 0x00504)
#define V3D_SRQPC (V3D_BASE + 0x00430)
#define V3D_SRQUA (V3D_BASE + 0x00434)
#define V3D_SRQCS (V3D_BASE + 0x0043c)
#define V3D_ERRSTAT (V3D_BASE + 0x00f20)

#define QPU_BARRIER_TIMEOUT_USEC 2000000u
#define QPU_BARRIER_LANES 16u
#define QPU_BARRIER_MAX_QPUS 16u
#define QPU_BARRIER_EXPECTED_QPUS 12u
#define QPU_BARRIER_VPM_URSV_4K 16u
#define QPU_BARRIER_SENTINEL 0xdeadbeefu
#define QPU_BARRIER_MAX_RUNS 8u
#define QPU_BARRIER_MAX_WARPS 12u
#define QPU_BARRIER_RESULT_GUARD_WORDS 16u
#define QPU_BARRIER_RESULT_GUARD_BASE 0xbaad0000u

#define QPU_BARRIER_MODE_SAME_SLICE_SMOKE 0u
#define QPU_BARRIER_MODE_CROSS_SLICE_0_1 1u
#define QPU_BARRIER_MODE_CROSS_SLICE_0_2 2u
#define QPU_BARRIER_MODE_FULL_BLOCK_STRESS 3u
#define QPU_BARRIER_MODE_TWO_BLOCK_PARTITION 4u

struct qpu_barrier_warp_result {
    uint32_t block_id;
    uint32_t logical_warp_id;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t physical_qpu_report[QPU_BARRIER_LANES];
    uint32_t all_seen_mask_by_lane[QPU_BARRIER_LANES];
    uint32_t always_seen_mask_by_lane[QPU_BARRIER_LANES];
    uint32_t mismatch_count_by_lane[QPU_BARRIER_LANES];
};

struct qpu_barrier_run_result {
    uint32_t run_id;
    uint32_t mode;
    uint32_t block_count;
    uint32_t warps_per_block;
    uint32_t total_requests;
    uint32_t iterations;
    uint32_t sem_base;
    uint32_t vpm_base_row[4];
    uint32_t vpm_rows;
    uint32_t qpu_set_mask;
    uint32_t expected_qpu_mask;
    uint32_t observed_qpu_mask;
    uint32_t mismatch_count;
    uint32_t timeout;
    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t first_bad_iter;
    uint32_t first_bad_block;
    uint32_t first_bad_reader_warp;
    uint32_t first_bad_peer_warp;
    uint32_t first_bad_lane;
    uint32_t first_bad_observed;
    uint32_t first_bad_expected;
    struct qpu_barrier_warp_result warp[QPU_BARRIER_MAX_WARPS];
};

struct qpu_barrier_results {
    uint32_t ident1;
    uint32_t vpmsz_field;
    uint32_t vpm_kib;
    uint32_t qpus_per_slice;
    uint32_t num_slices;
    uint32_t num_qpus;
    uint32_t num_semaphores;
    uint32_t vpmbase_written;
    uint32_t vpmbase_readback;
    uint32_t run_count;
    uint32_t srqcs_after_last_run;
    uint32_t timeouts;
    struct qpu_barrier_run_result runs[QPU_BARRIER_MAX_RUNS];
};

struct qpu_barrier_run_config {
    uint32_t mode;
    uint32_t block_count;
    uint32_t warps_per_block;
    uint32_t iterations;
    uint32_t allowed_qpu_mask;
};

struct qpu_barrier_device_state {
    uint32_t guard_before[QPU_BARRIER_RESULT_GUARD_WORDS];
    struct qpu_barrier_results results;
    uint32_t guard_after[QPU_BARRIER_RESULT_GUARD_WORDS];
};

static struct qpu_barrier_results results __attribute__((aligned(16)));

static uint32_t low_mask_local(uint32_t n) {
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

static uint32_t errstat_relevant_changed(uint32_t before, uint32_t after) {
    return (((before ^ after) & ERRSTAT_RELEVANT_MASK) != 0);
}

static const char *mode_name(uint32_t mode) {
    switch (mode) {
    case QPU_BARRIER_MODE_SAME_SLICE_SMOKE: return "same_slice_smoke";
    case QPU_BARRIER_MODE_CROSS_SLICE_0_1: return "cross_slice_0_1";
    case QPU_BARRIER_MODE_CROSS_SLICE_0_2: return "cross_slice_0_2";
    case QPU_BARRIER_MODE_FULL_BLOCK_STRESS: return "full_block_stress";
    case QPU_BARRIER_MODE_TWO_BLOCK_PARTITION: return "two_block_partition";
    default: return "unknown";
    }
}

static void decode_ident1(volatile struct qpu_barrier_results *res) {
    uint32_t ident1 = GET32(V3D_IDENT1);
    uint32_t vpmsz_field = (ident1 >> 28) & 0xfu;
    res->ident1 = ident1;
    res->vpmsz_field = vpmsz_field;
    res->vpm_kib = vpmsz_field ? vpmsz_field : 16u;
    res->num_semaphores = (ident1 >> 16) & 0xffu;
    res->qpus_per_slice = (ident1 >> 8) & 0xfu;
    res->num_slices = (ident1 >> 4) & 0xfu;
    res->num_qpus = res->qpus_per_slice * res->num_slices;
}

static void reserve_qpu_mask(uint32_t allowed_mask) {
    uint32_t sqrsv0 = 0;
    uint32_t sqrsv1 = 0;
    for (uint32_t q = 0; q < QPU_BARRIER_MAX_QPUS; q++) {
        uint32_t reserve_user_programs = ((allowed_mask & (1u << q)) == 0) ? 1u : 0u;
        if (q < 8u)
            sqrsv0 |= reserve_user_programs << (4u * q);
        else
            sqrsv1 |= reserve_user_programs << (4u * (q - 8u));
    }
    PUT32(V3D_SQRSV0, sqrsv0);
    PUT32(V3D_SQRSV1, sqrsv1);
}

static void clear_qpu_reservations(void) {
    PUT32(V3D_SQRSV0, 0);
    PUT32(V3D_SQRSV1, 0);
}

static void clear_scheduler_and_caches(void) {
    PUT32(V3D_DBCFG, 0);
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, 0xffffffffu);
    PUT32(V3D_L2CACTL, 1u << 2);
    PUT32(V3D_SLCACTL, 0xffffffffu);
    PUT32(V3D_SRQCS, (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16));
}

static int wait_for_completions(uint32_t expected) {
    uint32_t start = (uint32_t)timer_get_usec();
    while ((((GET32(V3D_SRQCS) >> 16) & 0xffu) != expected)) {
        uint32_t now = (uint32_t)timer_get_usec();
        if ((uint32_t)(now - start) > QPU_BARRIER_TIMEOUT_USEC)
            return -1;
    }
    return 0;
}

static void fill_vector(volatile uint32_t vec[QPU_BARRIER_LANES], uint32_t value) {
    for (uint32_t i = 0; i < QPU_BARRIER_LANES; i++)
        vec[i] = value;
}

static void clear_warp_result(volatile struct qpu_barrier_warp_result *warp,
                              uint32_t block_id,
                              uint32_t logical_warp_id) {
    warp->block_id = block_id;
    warp->logical_warp_id = logical_warp_id;
    warp->reserved0 = 0;
    warp->reserved1 = 0;
    fill_vector(warp->physical_qpu_report, QPU_BARRIER_SENTINEL);
    fill_vector(warp->all_seen_mask_by_lane, QPU_BARRIER_SENTINEL);
    fill_vector(warp->always_seen_mask_by_lane, QPU_BARRIER_SENTINEL);
    fill_vector(warp->mismatch_count_by_lane, QPU_BARRIER_SENTINEL);
}

static void prepare_run_result(volatile struct qpu_barrier_run_result *run,
                               uint32_t run_id,
                               const struct qpu_barrier_run_config *cfg) {
    memset((void *)run, 0, sizeof *run);
    run->run_id = run_id;
    run->mode = cfg->mode;
    run->block_count = cfg->block_count;
    run->warps_per_block = cfg->warps_per_block;
    run->total_requests = cfg->block_count * cfg->warps_per_block;
    run->iterations = cfg->iterations;
    run->vpm_base_row[0] = 0;
    run->vpm_base_row[1] = 16;
    run->vpm_base_row[2] = 32;
    run->vpm_base_row[3] = 48;
    run->vpm_rows = 16;
    run->qpu_set_mask = cfg->allowed_qpu_mask;
    run->expected_qpu_mask = cfg->allowed_qpu_mask;
    run->first_bad_iter = 0xffffffffu;
    run->first_bad_block = 0xffffffffu;
    run->first_bad_reader_warp = 0xffffffffu;
    run->first_bad_peer_warp = 0xffffffffu;
    run->first_bad_lane = 0xffffffffu;
    run->first_bad_observed = QPU_BARRIER_SENTINEL;
    run->first_bad_expected = QPU_BARRIER_SENTINEL;
    for (uint32_t block = 0; block < cfg->block_count; block++) {
        for (uint32_t warp = 0; warp < cfg->warps_per_block; warp++) {
            uint32_t request = block * cfg->warps_per_block + warp;
            clear_warp_result(&run->warp[request], block, warp);
        }
    }
}

static uint32_t validate_qpu_report(const struct qpu_barrier_warp_result *warp,
                                    uint32_t *qpu_out) {
    uint32_t qpu = warp->physical_qpu_report[0];
    uint32_t mismatches = 0;
    if (qpu >= QPU_BARRIER_MAX_QPUS)
        mismatches++;
    for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++) {
        if (warp->physical_qpu_report[lane] != qpu)
            mismatches++;
    }
    *qpu_out = qpu;
    return mismatches;
}

static uint32_t sum_mismatch_vector(const struct qpu_barrier_warp_result *warp) {
    uint32_t sum = 0;
    for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++)
        sum += warp->mismatch_count_by_lane[lane];
    return sum;
}

static uint32_t first_missing_peer(uint32_t mask, uint32_t expected_mask) {
    for (uint32_t peer = 0; peer < QPU_BARRIER_MAX_WARPS; peer++) {
        uint32_t bit = 1u << peer;
        if ((expected_mask & bit) && ((mask & bit) == 0))
            return peer;
    }
    return 0xffffffffu;
}

static uint32_t guard_word(uint32_t run_index, uint32_t word_index) {
    return QPU_BARRIER_RESULT_GUARD_BASE ^ (run_index << 8) ^ word_index;
}

static void fill_result_guard(uint32_t run_index, uint32_t guard[QPU_BARRIER_RESULT_GUARD_WORDS]) {
    for (uint32_t i = 0; i < QPU_BARRIER_RESULT_GUARD_WORDS; i++)
        guard[i] = guard_word(run_index, i);
}

static uint32_t verify_result_guard(uint32_t run_index,
                                    const uint32_t guard[QPU_BARRIER_RESULT_GUARD_WORDS]) {
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < QPU_BARRIER_RESULT_GUARD_WORDS; i++) {
        uint32_t expected = guard_word(run_index, i);
        uint32_t actual = guard[i];
        if (actual != expected) {
            if (mismatches < 8)
                printk("ERROR: barrier guard run=%d word=%d actual=0x%x expected=0x%x\n",
                       (int)run_index, (int)i, actual, expected);
            mismatches++;
        }
    }
    return mismatches;
}

static vc4_deviceptr_t device_state_offset(vc4_deviceptr_t base, uint32_t offset) {
    return (vc4_deviceptr_t)(base + offset);
}

static uint32_t run_result_offset(uint32_t run_index) {
    return (uint32_t)(offsetof(struct qpu_barrier_device_state, results) +
                      offsetof(struct qpu_barrier_results, runs) +
                      run_index * sizeof(struct qpu_barrier_run_result));
}

static int launch_run(struct vc4_program *program,
                      vc4_deviceptr_t state_dev,
                      struct qpu_barrier_device_state *state,
                      uint32_t run_index,
                      const struct qpu_barrier_run_config *cfg,
                      uint32_t *manual_launches) {
    struct qpu_barrier_run_result *run = &state->results.runs[run_index];
    uint32_t total_requests = cfg->block_count * cfg->warps_per_block;
    if (total_requests == 0 || total_requests > QPU_BARRIER_MAX_WARPS)
        return -1;

    prepare_run_result(run, run_index, cfg);
    uint32_t run_offset = run_result_offset(run_index);
    vc4_deviceptr_t run_dev = device_state_offset(state_dev, run_offset);
    vc4_deviceptr_t warp_dev =
        device_state_offset(run_dev, (uint32_t)offsetof(struct qpu_barrier_run_result, warp));

    if (vc4_m2_copy_htod(program, run_dev, run, sizeof(*run)) < 0)
        return -1;

    reserve_qpu_mask(cfg->allowed_qpu_mask);
    run->errstat_before = GET32(V3D_ERRSTAT);

    vc4_dim3 grid = vc4_m2_dim3(cfg->block_count, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(cfg->warps_per_block * QPU_BARRIER_LANES, 1u, 1u);
    (*manual_launches)++;
    int rc = qpu_barrier_syncthreads_launch(program, grid, block, cfg->mode, run_index,
                                            cfg->iterations, warp_dev, run_dev);
    run->errstat_after = GET32(V3D_ERRSTAT);
    clear_qpu_reservations();

    if (rc < 0) {
        run->timeout = 1;
        state->results.timeouts++;
        return -1;
    }
    if (vc4_m2_copy_dtoh(program, run, run_dev, sizeof(*run)) < 0)
        return -1;
    return 0;
}

static void analyze_runs(uint32_t *full_block_pass,
                         uint32_t *multi_block_pass,
                         uint32_t *qpu_mismatches,
                         uint32_t *data_mismatches,
                         uint32_t *timeouts,
                         uint32_t *errstat_relevant_changed_count,
                         uint32_t *same_slice_pass,
                         uint32_t *cross_slice_0_1_pass,
                         uint32_t *cross_slice_0_2_pass,
                         uint32_t *checked_warp_records,
                         uint32_t *checked_lane_records) {
    *full_block_pass = 0;
    *multi_block_pass = 0;
    *qpu_mismatches = 0;
    *data_mismatches = 0;
    *timeouts = 0;
    *errstat_relevant_changed_count = 0;
    *same_slice_pass = 0;
    *cross_slice_0_1_pass = 0;
    *cross_slice_0_2_pass = 0;
    *checked_warp_records = 0;
    *checked_lane_records = 0;

    printk("vc4_barrier_runs.csv:\n");
    printk("run,mode,blocks,warps_per_block,total_requests,iterations,expected_qpu_mask,observed_qpu_mask,mismatches,errstat_before,errstat_after,timeout\n");

    for (uint32_t i = 0; i < results.run_count; i++) {
        struct qpu_barrier_run_result *run = &results.runs[i];
        uint32_t expected_peer_mask = low_mask_local(run->warps_per_block);
        uint32_t run_data_mismatches = 0;
        uint32_t run_qpu_mismatches = 0;
        uint32_t observed_qpu_mask = 0;
        uint32_t run_err_changed = errstat_relevant_changed(run->errstat_before, run->errstat_after);

        if (run->timeout)
            (*timeouts)++;
        if (run_err_changed)
            (*errstat_relevant_changed_count)++;

        for (uint32_t request = 0; request < run->total_requests; request++) {
            const struct qpu_barrier_warp_result *warp = &run->warp[request];
            uint32_t qpu = 0xffffffffu;
            uint32_t warp_qpu_mismatches = validate_qpu_report(warp, &qpu);
            uint32_t warp_mismatch_sum = sum_mismatch_vector(warp);
            uint32_t all_seen0 = warp->all_seen_mask_by_lane[0];
            uint32_t always_seen0 = warp->always_seen_mask_by_lane[0];

            (*checked_warp_records)++;
            *checked_lane_records += QPU_BARRIER_LANES;
            run_qpu_mismatches += warp_qpu_mismatches;
            if (qpu < QPU_BARRIER_MAX_QPUS)
                observed_qpu_mask |= 1u << qpu;

            for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++) {
                if (warp->all_seen_mask_by_lane[lane] != expected_peer_mask)
                    run_data_mismatches++;
                if (warp->always_seen_mask_by_lane[lane] != expected_peer_mask)
                    run_data_mismatches++;
                if (warp->mismatch_count_by_lane[lane] != 0)
                    run_data_mismatches += warp->mismatch_count_by_lane[lane];
            }

            printk("BARRIER_WARP run=%d block=%d warp=%d qpu=%d all_seen=%x always_seen=%x mismatches=%d\n",
                   (int)run->run_id,
                   (int)warp->block_id,
                   (int)warp->logical_warp_id,
                   (int)qpu,
                   all_seen0,
                   always_seen0,
                   (int)warp_mismatch_sum);

            if (run->first_bad_iter == 0xffffffffu) {
                uint32_t missing = first_missing_peer(always_seen0, expected_peer_mask);
                if (warp_qpu_mismatches || warp_mismatch_sum ||
                    all_seen0 != expected_peer_mask ||
                    always_seen0 != expected_peer_mask) {
                    run->first_bad_block = warp->block_id;
                    run->first_bad_reader_warp = warp->logical_warp_id;
                    run->first_bad_peer_warp = missing;
                    run->first_bad_lane = 0;
                    run->first_bad_observed = all_seen0;
                    run->first_bad_expected = expected_peer_mask;
                }
            }
        }

        run->observed_qpu_mask = observed_qpu_mask;
        if (observed_qpu_mask != run->expected_qpu_mask)
            run_qpu_mismatches++;
        run->mismatch_count = run_data_mismatches;
        *qpu_mismatches += run_qpu_mismatches;
        *data_mismatches += run_data_mismatches;

        printk("%d,%s,%d,%d,%d,%d,%x,%x,%d,%x,%x,%d\n",
               (int)run->run_id,
               mode_name(run->mode),
               (int)run->block_count,
               (int)run->warps_per_block,
               (int)run->total_requests,
               (int)run->iterations,
               run->expected_qpu_mask,
               run->observed_qpu_mask,
               (int)run_data_mismatches,
               run->errstat_before,
               run->errstat_after,
               (int)run->timeout);

        uint32_t pass = (run_data_mismatches == 0 &&
                         run_qpu_mismatches == 0 &&
                         run->timeout == 0 &&
                         run_err_changed == 0);
        if (run->mode == QPU_BARRIER_MODE_SAME_SLICE_SMOKE)
            *same_slice_pass = pass;
        if (run->mode == QPU_BARRIER_MODE_CROSS_SLICE_0_1)
            *cross_slice_0_1_pass = pass;
        if (run->mode == QPU_BARRIER_MODE_CROSS_SLICE_0_2)
            *cross_slice_0_2_pass = pass;
        if (run->mode == QPU_BARRIER_MODE_FULL_BLOCK_STRESS)
            *full_block_pass = pass;
        if (run->mode == QPU_BARRIER_MODE_TWO_BLOCK_PARTITION)
            *multi_block_pass = pass;
    }
}

void notmain(void) {
    int launch_failures = 0;
    struct vc4_program *program = 0;
    int rc = vc4_program_create(&program, 0);
    if (rc < 0) {
        printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=FAIL runs=0 checked_warp_records=0 checked_lane_records=0 same_slice_pass=0 cross_slice_0_1_pass=0 cross_slice_0_2_pass=0 full_block_pass=0 multi_block_pass=0 qpu_mismatches=0 data_mismatches=0 guard_mismatches=0 timeouts=0 invalid_topology=1 errstat_relevant_changed=0 launch_failures=1 manual_launches=0 code_words=0\n");
        return;
    }

    vc4_deviceptr_t state_dev = 0;
    if (vc4_m2_malloc(program, &state_dev, sizeof(struct qpu_barrier_device_state)) < 0) {
        printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=FAIL runs=0 checked_warp_records=0 checked_lane_records=0 same_slice_pass=0 cross_slice_0_1_pass=0 cross_slice_0_2_pass=0 full_block_pass=0 multi_block_pass=0 qpu_mismatches=0 data_mismatches=0 guard_mismatches=0 timeouts=0 invalid_topology=1 errstat_relevant_changed=0 launch_failures=1 manual_launches=0 code_words=0\n");
        vc4_program_destroy(program);
        return;
    }

    struct qpu_barrier_device_state device_state;
    struct qpu_barrier_device_state device_result;
    memset(&device_state, 0, sizeof(device_state));
    memset(&device_result, 0, sizeof(device_result));
    fill_result_guard(0, device_state.guard_before);
    fill_result_guard(1, device_state.guard_after);

    decode_ident1(&device_state.results);
    device_state.results.vpmbase_written = QPU_BARRIER_VPM_URSV_4K;
    PUT32(V3D_VPMBASE, QPU_BARRIER_VPM_URSV_4K);
    device_state.results.vpmbase_readback = GET32(V3D_VPMBASE);
    if (vc4_m2_copy_htod(program, state_dev, &device_state, sizeof(device_state)) < 0) {
        printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=FAIL runs=0 checked_warp_records=0 checked_lane_records=0 same_slice_pass=0 cross_slice_0_1_pass=0 cross_slice_0_2_pass=0 full_block_pass=0 multi_block_pass=0 qpu_mismatches=0 data_mismatches=0 guard_mismatches=0 timeouts=0 invalid_topology=1 errstat_relevant_changed=0 launch_failures=1 manual_launches=0 code_words=0\n");
        vc4Free(program, state_dev);
        vc4_program_destroy(program);
        return;
    }

    uint32_t invalid_topology = 0;
    if (device_state.results.num_qpus < QPU_BARRIER_EXPECTED_QPUS)
        invalid_topology = 1;
    if (device_state.results.qpus_per_slice != 4u)
        invalid_topology = 1;
    if (device_state.results.num_slices < 3u)
        invalid_topology = 1;
    if (device_state.results.num_semaphores < 16u)
        invalid_topology = 1;
    if ((device_state.results.vpmbase_readback & 0x1fu) != QPU_BARRIER_VPM_URSV_4K)
        invalid_topology = 1;

    uint32_t qps = device_state.results.qpus_per_slice;
    uint32_t cross1 = qps;
    uint32_t cross2 = 2u * qps;
    struct qpu_barrier_run_config configs[5];
    configs[0] = (struct qpu_barrier_run_config){QPU_BARRIER_MODE_SAME_SLICE_SMOKE, 1u, 2u, 4u, (1u << 0) | (1u << 1)};
    configs[1] = (struct qpu_barrier_run_config){QPU_BARRIER_MODE_CROSS_SLICE_0_1, 1u, 2u, 4u, (1u << 0) | (1u << cross1)};
    configs[2] = (struct qpu_barrier_run_config){QPU_BARRIER_MODE_CROSS_SLICE_0_2, 1u, 2u, 4u, (1u << 0) | (1u << cross2)};
    configs[3] = (struct qpu_barrier_run_config){QPU_BARRIER_MODE_FULL_BLOCK_STRESS, 1u, 12u, 64u, low_mask_local(12u)};
    configs[4] = (struct qpu_barrier_run_config){QPU_BARRIER_MODE_TWO_BLOCK_PARTITION, 2u, 4u, 32u, low_mask_local(4u)};

    int start = timer_get_usec();
    device_state.results.run_count = 5;
    uint32_t manual_launches = 0;
    if (!invalid_topology) {
        for (uint32_t i = 0; i < device_state.results.run_count; i++) {
            if (launch_run(program, state_dev, &device_state, i, &configs[i], &manual_launches) < 0)
                launch_failures++;
        }
    }
    int elapsed = timer_get_usec() - start;

    if (vc4_m2_copy_htod(program,
                         device_state_offset(state_dev, (uint32_t)offsetof(struct qpu_barrier_device_state, results)),
                         &device_state.results,
                         sizeof(device_state.results)) < 0 ||
        vc4_m2_copy_dtoh(program, &device_result, state_dev, sizeof(device_result)) < 0) {
        launch_failures++;
    }

    uint32_t guard_mismatches =
        verify_result_guard(0, device_result.guard_before) +
        verify_result_guard(1, device_result.guard_after);
    memcpy(&results, &device_result.results, sizeof(results));

    uint32_t full_block_pass = 0;
    uint32_t multi_block_pass = 0;
    uint32_t qpu_mismatches = 0;
    uint32_t data_mismatches = 0;
    uint32_t timeouts = 0;
    uint32_t errstat_relevant_changed_count = 0;
    uint32_t same_slice_pass = 0;
    uint32_t cross_slice_0_1_pass = 0;
    uint32_t cross_slice_0_2_pass = 0;
    uint32_t checked_warp_records = 0;
    uint32_t checked_lane_records = 0;

    analyze_runs(&full_block_pass,
                 &multi_block_pass,
                 &qpu_mismatches,
                 &data_mismatches,
                 &timeouts,
                 &errstat_relevant_changed_count,
                 &same_slice_pass,
                 &cross_slice_0_1_pass,
                 &cross_slice_0_2_pass,
                 &checked_warp_records,
                 &checked_lane_records);

    uint32_t code_words = qpu_barrier_syncthreads_runtime_code_uploads();

    uint32_t status_pass = (launch_failures == 0 &&
                            invalid_topology == 0 &&
                            results.run_count == 5u &&
                            manual_launches == 5u &&
                            code_words == 1u &&
                            checked_warp_records == 26u &&
                            checked_lane_records == 416u &&
                            same_slice_pass &&
                            cross_slice_0_1_pass &&
                            cross_slice_0_2_pass &&
                            full_block_pass &&
                            multi_block_pass &&
                            qpu_mismatches == 0 &&
                            data_mismatches == 0 &&
                            guard_mismatches == 0 &&
                            timeouts == 0 &&
                            errstat_relevant_changed_count == 0);

    printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=%s runs=%d checked_warp_records=%d checked_lane_records=%d same_slice_pass=%d cross_slice_0_1_pass=%d cross_slice_0_2_pass=%d full_block_pass=%d multi_block_pass=%d qpu_mismatches=%d data_mismatches=%d guard_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d launch_failures=%d manual_launches=%d code_words=%d elapsed_usec=%d\n",
           status_pass ? "PASS" : "FAIL",
           (int)results.run_count,
           (int)checked_warp_records,
           (int)checked_lane_records,
           (int)same_slice_pass,
           (int)cross_slice_0_1_pass,
           (int)cross_slice_0_2_pass,
           (int)full_block_pass,
           (int)multi_block_pass,
           (int)qpu_mismatches,
           (int)data_mismatches,
           (int)guard_mismatches,
           (int)timeouts,
           (int)invalid_topology,
           (int)errstat_relevant_changed_count,
           launch_failures,
           (int)manual_launches,
           (int)code_words,
           elapsed);

    clear_qpu_reservations();
    vc4Free(program, state_dev);
    vc4_program_destroy(program);
}
