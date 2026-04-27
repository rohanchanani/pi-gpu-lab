#include "rpi.h"
#include "qpu_barrier_syncthreads_launch.h"

#define ERRSTAT_RELEVANT_MASK 0x0000efffu /* exclude VCD idle bit 12 */

static struct qpu_barrier_results results
    __attribute__((aligned(16)));

static uint32_t low_mask_local(uint32_t n)
{
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

static uint32_t tag(uint32_t run_id,
                    uint32_t block_id,
                    uint32_t iter,
                    uint32_t logical_warp_id,
                    uint32_t lane)
{
    return 0xb0000000u |
           ((run_id & 0x0fu) << 24) |
           ((block_id & 0x0fu) << 20) |
           ((iter & 0xffu) << 12) |
           ((logical_warp_id & 0x0fu) << 8) |
           (lane & 0x0fu);
}

static uint32_t errstat_relevant_changed(uint32_t before, uint32_t after)
{
    return (((before ^ after) & ERRSTAT_RELEVANT_MASK) != 0);
}

static const char *mode_name(uint32_t mode)
{
    switch (mode)
    {
    case QPU_BARRIER_MODE_SAME_SLICE_SMOKE:
        return "same_slice_smoke";
    case QPU_BARRIER_MODE_CROSS_SLICE_0_1:
        return "cross_slice_0_1";
    case QPU_BARRIER_MODE_CROSS_SLICE_0_2:
        return "cross_slice_0_2";
    case QPU_BARRIER_MODE_FULL_BLOCK_STRESS:
        return "full_block_stress";
    case QPU_BARRIER_MODE_TWO_BLOCK_PARTITION:
        return "two_block_partition";
    default:
        return "unknown";
    }
}

static void print_topology_json(void)
{
    printk("vc4_barrier_topology.json:\n");
    printk("{\n");
    printk("  ident1: %x,\n", results.ident1);
    printk("  vpmsz_field: %d,\n", (int)results.vpmsz_field);
    printk("  vpm_kib: %d,\n", (int)results.vpm_kib);
    printk("  qpus_per_slice: %d,\n", (int)results.qpus_per_slice);
    printk("  num_slices: %d,\n", (int)results.num_slices);
    printk("  num_qpus: %d,\n", (int)results.num_qpus);
    printk("  num_semaphores: %d,\n", (int)results.num_semaphores);
    printk("  vpmbase_written: %d,\n", (int)results.vpmbase_written);
    printk("  vpmbase_readback: %x\n", results.vpmbase_readback);
    printk("}\n");
}

static uint32_t validate_qpu_report(
    const struct qpu_barrier_warp_result *warp,
    uint32_t *qpu_out)
{
    uint32_t qpu = warp->physical_qpu_report[0];
    uint32_t mismatches = 0;

    if (qpu >= QPU_BARRIER_MAX_QPUS)
        mismatches++;

    for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++)
    {
        if (warp->physical_qpu_report[lane] != qpu)
            mismatches++;
    }

    *qpu_out = qpu;
    return mismatches;
}

static uint32_t sum_mismatch_vector(
    const struct qpu_barrier_warp_result *warp)
{
    uint32_t sum = 0;
    for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++)
        sum += warp->mismatch_count_by_lane[lane];
    return sum;
}

static uint32_t first_missing_peer(uint32_t mask, uint32_t expected_mask)
{
    for (uint32_t peer = 0; peer < QPU_BARRIER_MAX_WARPS; peer++)
    {
        uint32_t bit = 1u << peer;
        if ((expected_mask & bit) && ((mask & bit) == 0))
            return peer;
    }
    return 0xffffffffu;
}

static void analyze_runs(uint32_t *full_block_pass,
                         uint32_t *multi_block_pass,
                         uint32_t *qpu_mismatches,
                         uint32_t *data_mismatches,
                         uint32_t *timeouts,
                         uint32_t *errstat_relevant_changed_count,
                         uint32_t *same_slice_pass,
                         uint32_t *cross_slice_0_1_pass,
                         uint32_t *cross_slice_0_2_pass)
{
    *full_block_pass = 0;
    *multi_block_pass = 0;
    *qpu_mismatches = 0;
    *data_mismatches = 0;
    *timeouts = 0;
    *errstat_relevant_changed_count = 0;
    *same_slice_pass = 0;
    *cross_slice_0_1_pass = 0;
    *cross_slice_0_2_pass = 0;

    printk("vc4_barrier_runs.csv:\n");
    printk("run,mode,blocks,warps_per_block,total_requests,iterations,expected_qpu_mask,observed_qpu_mask,mismatches,errstat_before,errstat_after,timeout\n");

    for (uint32_t i = 0; i < results.run_count; i++)
    {
        struct qpu_barrier_run_result *run = &results.runs[i];
        uint32_t expected_peer_mask = low_mask_local(run->warps_per_block);
        uint32_t run_data_mismatches = 0;
        uint32_t run_qpu_mismatches = 0;
        uint32_t observed_qpu_mask = 0;
        uint32_t run_err_changed = errstat_relevant_changed(run->errstat_before,
                                                            run->errstat_after);

        if (run->timeout)
            (*timeouts)++;
        if (run_err_changed)
            (*errstat_relevant_changed_count)++;

        for (uint32_t request = 0; request < run->total_requests; request++)
        {
            const struct qpu_barrier_warp_result *warp = &run->warp[request];
            uint32_t qpu = 0xffffffffu;
            uint32_t warp_qpu_mismatches = validate_qpu_report(warp, &qpu);
            uint32_t warp_mismatch_sum = sum_mismatch_vector(warp);
            uint32_t all_seen0 = warp->all_seen_mask_by_lane[0];
            uint32_t always_seen0 = warp->always_seen_mask_by_lane[0];

            run_qpu_mismatches += warp_qpu_mismatches;
            if (qpu < QPU_BARRIER_MAX_QPUS)
                observed_qpu_mask |= 1u << qpu;

            for (uint32_t lane = 0; lane < QPU_BARRIER_LANES; lane++)
            {
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

            if (run->first_bad_iter == 0xffffffffu)
            {
                uint32_t missing = first_missing_peer(always_seen0,
                                                      expected_peer_mask);
                if (warp_qpu_mismatches || warp_mismatch_sum ||
                    all_seen0 != expected_peer_mask ||
                    always_seen0 != expected_peer_mask)
                {
                    run->first_bad_iter = 0xffffffffu;
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

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    printk("Running VC4 qpu_barrier_syncthreads reference bundle...\n");
    int start = timer_get_usec();
    int launch_rc = qpu_barrier_syncthreads_launch(&rt, &results);
    int end = timer_get_usec();
    int elapsed = end - start;

    print_topology_json();

    uint32_t invalid_topology = 0;
    if (results.num_qpus < QPU_BARRIER_EXPECTED_QPUS)
        invalid_topology = 1;
    if (results.qpus_per_slice != 4u)
        invalid_topology = 1;
    if (results.num_slices < 3u)
        invalid_topology = 1;
    if (results.num_semaphores < 16u)
        invalid_topology = 1;
    if ((results.vpmbase_readback & 0x1fu) != QPU_BARRIER_VPM_URSV_4K)
        invalid_topology = 1;

    uint32_t full_block_pass = 0;
    uint32_t multi_block_pass = 0;
    uint32_t qpu_mismatches = 0;
    uint32_t data_mismatches = 0;
    uint32_t timeouts = 0;
    uint32_t errstat_relevant_changed_count = 0;
    uint32_t same_slice_pass = 0;
    uint32_t cross_slice_0_1_pass = 0;
    uint32_t cross_slice_0_2_pass = 0;

    analyze_runs(&full_block_pass,
                 &multi_block_pass,
                 &qpu_mismatches,
                 &data_mismatches,
                 &timeouts,
                 &errstat_relevant_changed_count,
                 &same_slice_pass,
                 &cross_slice_0_1_pass,
                 &cross_slice_0_2_pass);

    uint32_t status_pass = (launch_rc == 0 &&
                            invalid_topology == 0 &&
                            results.run_count == 5u &&
                            same_slice_pass &&
                            cross_slice_0_1_pass &&
                            cross_slice_0_2_pass &&
                            full_block_pass &&
                            multi_block_pass &&
                            qpu_mismatches == 0 &&
                            data_mismatches == 0 &&
                            timeouts == 0 &&
                            errstat_relevant_changed_count == 0);

    if (status_pass)
    {
        printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=PASS runs=%d same_slice_pass=%d cross_slice_0_1_pass=%d cross_slice_0_2_pass=%d full_block_pass=%d multi_block_pass=%d qpu_mismatches=%d data_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d elapsed_usec=%d\n",
               (int)results.run_count,
               (int)same_slice_pass,
               (int)cross_slice_0_1_pass,
               (int)cross_slice_0_2_pass,
               (int)full_block_pass,
               (int)multi_block_pass,
               (int)qpu_mismatches,
               (int)data_mismatches,
               (int)timeouts,
               (int)invalid_topology,
               (int)errstat_relevant_changed_count,
               elapsed);
    }
    else
    {
        struct qpu_barrier_run_result *bad = &results.runs[0];
        for (uint32_t i = 0; i < results.run_count; i++)
        {
            if (results.runs[i].timeout || results.runs[i].mismatch_count ||
                results.runs[i].observed_qpu_mask != results.runs[i].expected_qpu_mask ||
                errstat_relevant_changed(results.runs[i].errstat_before,
                                         results.runs[i].errstat_after))
            {
                bad = &results.runs[i];
                break;
            }
        }

        printk("VC4_TEST_RESULT name=qpu_barrier_syncthreads status=FAIL runs=%d launch_rc=%d first_bad_run=%d first_bad_iter=%x first_bad_block=%x first_bad_reader_warp=%x first_bad_peer_warp=%x first_bad_lane=%x observed=%x expected=%x same_slice_pass=%d cross_slice_0_1_pass=%d cross_slice_0_2_pass=%d full_block_pass=%d multi_block_pass=%d qpu_mismatches=%d data_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d elapsed_usec=%d\n",
               (int)results.run_count,
               launch_rc,
               (int)bad->run_id,
               bad->first_bad_iter,
               bad->first_bad_block,
               bad->first_bad_reader_warp,
               bad->first_bad_peer_warp,
               bad->first_bad_lane,
               bad->first_bad_observed,
               bad->first_bad_expected,
               (int)same_slice_pass,
               (int)cross_slice_0_1_pass,
               (int)cross_slice_0_2_pass,
               (int)full_block_pass,
               (int)multi_block_pass,
               (int)qpu_mismatches,
               (int)data_mismatches,
               (int)timeouts,
               (int)invalid_topology,
               (int)errstat_relevant_changed_count,
               elapsed);
    }

    vc4_runtime_shutdown(&rt);
}
