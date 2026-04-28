#include "rpi.h"
#include "block_reduce_sum_launch.h"

#define BLOCK_REDUCE_SUM_EPSILON 0.0005f
#define CHECKSUM_SCALE 4096.0f

enum {
BRS_CASES = 6,
BRS_MAX_BLOCKS = 3,
BRS_MAX_VALUES_PER_BLOCK = 192,
BRS_MAX_INPUT = BRS_MAX_BLOCKS * BRS_MAX_VALUES_PER_BLOCK,
BRS_MAX_OUTPUT = BRS_MAX_BLOCKS * BLOCK_REDUCE_SUM_RESULT_LANES
};

struct block_reduce_sum_case {
uint32_t blocks;
uint32_t warps_per_block;
uint32_t values_per_block;
};

static const struct block_reduce_sum_case cases[BRS_CASES] = {
{1u, 1u, 1u},
{1u, 2u, 31u},
{1u, 4u, 64u},
{1u, 12u, 192u},
{1u, 12u, 173u},
{3u, 4u, 59u},
};

static float input_values[BRS_MAX_INPUT];
static float out_values[BRS_MAX_OUTPUT];
static float expected_values[BRS_MAX_OUTPUT];

static float absf_local(float value)
{
return value < 0.0f ? -value : value;
}

static float make_input_value(uint32_t i, uint32_t case_id)
{
int raw = (int)((i * 5u + case_id * 7u + 3u) % 17u) - 8;
return (float)raw * 0.125f;
}

static void fill_inputs(uint32_t case_id, uint32_t blocks, uint32_t values_per_block)
{
for (uint32_t i = 0; i < BRS_MAX_INPUT; i++)
input_values[i] = 0.0f;
for (uint32_t i = 0; i < BRS_MAX_OUTPUT; i++) {
out_values[i] = BLOCK_REDUCE_SUM_SENTINEL;
expected_values[i] = BLOCK_REDUCE_SUM_SENTINEL;
}

for (uint32_t block = 0; block < blocks; block++) {
    for (uint32_t i = 0; i < values_per_block; i++) {
        uint32_t logical = block * values_per_block + i;
        input_values[logical] = make_input_value(logical, case_id);
    }
}

}

static void run_cpu_reference(uint32_t blocks, uint32_t values_per_block)
{
for (uint32_t block = 0; block < blocks; block++) {
float sum = 0.0f;
for (uint32_t i = 0; i < values_per_block; i++)
sum += input_values[block * values_per_block + i];

    for (uint32_t lane = 0; lane < BLOCK_REDUCE_SUM_RESULT_LANES; lane++)
        expected_values[block * BLOCK_REDUCE_SUM_RESULT_LANES + lane] = sum;
}

}

static void verify_results(
uint32_t case_id,
uint32_t blocks,
int *mismatches,
float *max_abs_diff)
{
*mismatches = 0;
*max_abs_diff = 0.0f;

uint32_t count = blocks * BLOCK_REDUCE_SUM_RESULT_LANES;
for (uint32_t i = 0; i < count; i++) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
        *max_abs_diff = abs_diff;

    if (abs_diff > BLOCK_REDUCE_SUM_EPSILON) {
        if (*mismatches < 8) {
            uint32_t block = i / BLOCK_REDUCE_SUM_RESULT_LANES;
            uint32_t lane = i % BLOCK_REDUCE_SUM_RESULT_LANES;
            printk("BLOCK_REDUCE_BLOCK case=%d block=%d lane=%d gpu=%f cpu=%f diff=%f\n",
                   (int)case_id,
                   (int)block,
                   (int)lane,
                   out_values[i],
                   expected_values[i],
                   diff);
        }
        (*mismatches)++;
    }
}

}

static int scaled_checksum(const float *values, uint32_t count)
{
int checksum = 0;
for (uint32_t i = 0; i < count; i++)
checksum += (int)(values[i] * CHECKSUM_SCALE);
return checksum;
}

void notmain(void)
{
struct vc4_runtime rt;
struct block_reduce_sum_state state;

if (vc4_runtime_init(&rt) < 0)
    panic("Failed to initialize VC4 runtime");

uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
uint32_t lane_width = vc4_runtime_lane_width();

if (active_qpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)active_qpus);
if (lane_width != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)lane_width);

if (block_reduce_sum_prepare(&rt,
                             &state,
                             BRS_MAX_BLOCKS,
                             BRS_MAX_VALUES_PER_BLOCK) < 0)
    panic("block_reduce_sum runtime setup failed");

printk("BLOCK_REDUCE_SUM_RUNTIME_SETUP max_blocks=%d max_values_per_block=%d allocations=%d ident1=%x vpmbase=%x\n",
       BRS_MAX_BLOCKS,
       BRS_MAX_VALUES_PER_BLOCK,
       (int)block_reduce_sum_runtime_allocations(&state),
       (unsigned)block_reduce_sum_runtime_ident1(&state),
       (unsigned)block_reduce_sum_runtime_vpmbase_readback(&state));

int total_mismatches = 0;
int launch_failures = 0;
int checksum_accum = 0;
float max_abs_diff_overall = 0.0f;
uint32_t full_block_pass = 0;
uint32_t tail_pass = 0;
uint32_t multi_block_pass = 0;

int start = timer_get_usec();

for (uint32_t case_index = 0; case_index < BRS_CASES; case_index++) {
    uint32_t blocks = cases[case_index].blocks;
    uint32_t warps_per_block = cases[case_index].warps_per_block;
    uint32_t values_per_block = cases[case_index].values_per_block;

    fill_inputs(case_index, blocks, values_per_block);
    run_cpu_reference(blocks, values_per_block);

    if (block_reduce_sum_launch(&state,
                                input_values,
                                out_values,
                                blocks,
                                values_per_block,
                                warps_per_block) < 0) {
        printk("ERROR: block_reduce_sum launch failed case=%d blocks=%d warps_per_block=%d values_per_block=%d\n",
               (int)case_index,
               (int)blocks,
               (int)warps_per_block,
               (int)values_per_block);
        launch_failures++;
        continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(case_index, blocks, &mismatches, &max_abs_diff);
    int checksum = scaled_checksum(out_values,
                                   blocks * BLOCK_REDUCE_SUM_RESULT_LANES);
    int expected_checksum = scaled_checksum(expected_values,
                                            blocks * BLOCK_REDUCE_SUM_RESULT_LANES);
    if (checksum != expected_checksum) {
        printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n",
               (int)case_index,
               checksum,
               expected_checksum);
        mismatches++;
    }

    if (max_abs_diff > max_abs_diff_overall)
        max_abs_diff_overall = max_abs_diff;

    total_mismatches += mismatches;
    checksum_accum += checksum;

    if (case_index == 3u && mismatches == 0)
        full_block_pass = 1;
    if (case_index == 4u && mismatches == 0)
        tail_pass = 1;
    if (case_index == 5u && mismatches == 0)
        multi_block_pass = 1;

    printk("BLOCK_REDUCE_SUM_CASE case=%d blocks=%d warps_per_block=%d values_per_block=%d mismatches=%d checksum=%d max_abs_diff=%f launches=%d tile_waves=%d allocations=%d\n",
           (int)case_index,
           (int)blocks,
           (int)warps_per_block,
           (int)values_per_block,
           mismatches,
           checksum,
           max_abs_diff,
           (int)block_reduce_sum_runtime_launches(&state),
           (int)block_reduce_sum_runtime_tile_waves(&state),
           (int)block_reduce_sum_runtime_allocations(&state));
}

int end = timer_get_usec();
int elapsed = end - start;

uint32_t runtime_allocations =
    block_reduce_sum_runtime_allocations(&state);
uint32_t runtime_launches =
    block_reduce_sum_runtime_launches(&state);
uint32_t runtime_timeouts =
    block_reduce_sum_runtime_timeouts(&state);
uint32_t errstat_relevant_changed =
    block_reduce_sum_runtime_errstat_relevant_changed(&state);

const char *status =
    (total_mismatches == 0 &&
     launch_failures == 0 &&
     full_block_pass == 1u &&
     tail_pass == 1u &&
     multi_block_pass == 1u &&
     runtime_allocations == 1u &&
     runtime_launches == BRS_CASES &&
     runtime_timeouts == 0u &&
     errstat_relevant_changed == 0u &&
     max_abs_diff_overall <= BLOCK_REDUCE_SUM_EPSILON) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=block_reduce_sum status=%s cases=%d total_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_values_per_block=%d full_block_pass=%d tail_pass=%d multi_block_pass=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_tile_waves=%d timeouts=%d errstat_relevant_changed=%d srqcs_after_last_wave=%x elapsed_usec=%d\n",
       status,
       BRS_CASES,
       total_mismatches,
       launch_failures,
       (int)active_qpus,
       (int)lane_width,
       BRS_MAX_VALUES_PER_BLOCK,
       (int)full_block_pass,
       (int)tail_pass,
       (int)multi_block_pass,
       checksum_accum,
       max_abs_diff_overall,
       (int)runtime_allocations,
       (int)runtime_launches,
       (int)block_reduce_sum_runtime_tile_waves(&state),
       (int)runtime_timeouts,
       (int)errstat_relevant_changed,
       (unsigned)block_reduce_sum_runtime_srqcs_after_last_wave(&state),
       elapsed);

vc4_runtime_shutdown(&rt);

}

