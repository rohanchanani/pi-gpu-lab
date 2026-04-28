#include "rpi.h"
#include "global_store_coalesced_multi_launch.h"

#define GLOBAL_STORE_COHERENT_MAX_N 1000u
#define GLOBAL_STORE_COHERENT_GUARD GLOBAL_STORE_COALESCED_MULTI_GUARD_WORDS
#define GLOBAL_STORE_COHERENT_BUFFER_N (GLOBAL_STORE_COHERENT_MAX_N + GLOBAL_STORE_COHERENT_GUARD)

static uint32_t out_values[GLOBAL_STORE_COHERENT_BUFFER_N];

static const uint32_t test_sizes[] = {
0u,
1u,
2u,
15u,
16u,
17u,
31u,
32u,
33u,
191u,
192u,
193u,
255u,
256u,
257u,
767u,
768u,
769u,
1000u,
};

static uint32_t expected_value(uint32_t case_id, uint32_t i)
{
return 0x51000000u |
((case_id & 0xffu) << 16) |
(i & 0xffffu);
}

static void fill_host_buffer(uint32_t n)
{
for (uint32_t i = 0; i < GLOBAL_STORE_COHERENT_BUFFER_N; i++)
out_values[i] = QPU_STORE_SENTINEL;

for (uint32_t i = n; i < n + GLOBAL_STORE_COHERENT_GUARD &&
                     i < GLOBAL_STORE_COHERENT_BUFFER_N; i++)
    out_values[i] = QPU_STORE_SENTINEL;

}

static int verify_results(uint32_t case_id, uint32_t n)
{
int mismatches = 0;

for (uint32_t i = 0; i < n; i++) {
    uint32_t expected = expected_value(case_id, i);
    if (out_values[i] != expected) {
        if (mismatches < 8) {
            printk("ERROR: case=%d i=%d gpu=%x expected=%x\n",
                   (int)case_id,
                   (int)i,
                   out_values[i],
                   expected);
        }
        mismatches++;
    }
}

return mismatches;

}

static int verify_sentinel_tail(uint32_t n)
{
int mismatches = 0;

for (uint32_t i = n; i < n + GLOBAL_STORE_COHERENT_GUARD &&
                     i < GLOBAL_STORE_COHERENT_BUFFER_N; i++) {
    if (out_values[i] != QPU_STORE_SENTINEL) {
        if (mismatches < 8) {
            printk("ERROR: sentinel changed i=%d value=%x expected=%x\n",
                   (int)i,
                   out_values[i],
                   QPU_STORE_SENTINEL);
        }
        mismatches++;
    }
}

return mismatches;

}

static int checksum_u32(const uint32_t *values, uint32_t n)
{
int checksum = 0;
for (uint32_t i = 0; i < n; i++)
checksum += (int)(values[i] & 0xffffu);
return checksum;
}

void notmain(void)
{
struct vc4_runtime rt;
struct global_store_coalesced_multi_state state;

if (vc4_runtime_init(&rt) < 0)
    panic("Failed to initialize VC4 runtime");

uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
uint32_t lane_width = vc4_runtime_lane_width();

if (active_qpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)active_qpus);
if (lane_width != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)lane_width);

if (global_store_coalesced_multi_prepare(&rt,
                                         &state,
                                         GLOBAL_STORE_COHERENT_MAX_N) < 0)
    panic("global_store_coalesced_multi runtime setup failed");

printk("GLOBAL_STORE_COHERENT_RUNTIME_SETUP max_n=%d allocations=%d active_qpus=%d lanes=%d\n",
       GLOBAL_STORE_COHERENT_MAX_N,
       (int)global_store_coalesced_multi_runtime_allocations(&state),
       (int)active_qpus,
       (int)lane_width);

const uint32_t case_count = sizeof(test_sizes) / sizeof(test_sizes[0]);
int total_mismatches = 0;
int sentinel_mismatches = 0;
int launch_failures = 0;
int checksum_accum = 0;

int start = timer_get_usec();

for (uint32_t case_index = 0; case_index < case_count; case_index++) {
    uint32_t n = test_sizes[case_index];
    uint32_t case_id = case_index;

    if (n > GLOBAL_STORE_COHERENT_MAX_N)
        panic("test n exceeds max_n: %d", (int)n);

    fill_host_buffer(n);

    if (global_store_coalesced_multi_launch(&state,
                                            out_values,
                                            n,
                                            case_id) < 0) {
        printk("ERROR: global_store_coalesced_multi launch failed case=%d n=%d\n",
               (int)case_id,
               (int)n);
        launch_failures++;
        continue;
    }

    int mismatches = verify_results(case_id, n);
    int case_sentinel_mismatches = verify_sentinel_tail(n);
    int checksum = checksum_u32(out_values, n);

    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;

    printk("GLOBAL_STORE_COHERENT_CASE case=%d n=%d qpus=%d lanes=%d mismatches=%d sentinel_mismatches=%d checksum=%d launches=%d allocations=%d\n",
           (int)case_id,
           (int)n,
           (int)active_qpus,
           (int)lane_width,
           mismatches,
           case_sentinel_mismatches,
           checksum,
           (int)global_store_coalesced_multi_runtime_launches(&state),
           (int)global_store_coalesced_multi_runtime_allocations(&state));
}

int end = timer_get_usec();
int elapsed = end - start;

uint32_t runtime_allocations =
    global_store_coalesced_multi_runtime_allocations(&state);
uint32_t runtime_launches =
    global_store_coalesced_multi_runtime_launches(&state);
uint32_t runtime_capacity =
    global_store_coalesced_multi_runtime_capacity(&state);

const char *status =
    (total_mismatches == 0 &&
     sentinel_mismatches == 0 &&
     launch_failures == 0 &&
     runtime_allocations == 1u &&
     runtime_launches == case_count &&
     runtime_capacity == GLOBAL_STORE_COHERENT_MAX_N) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=global_store_coalesced_multi status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
       status,
       (int)case_count,
       total_mismatches,
       sentinel_mismatches,
       launch_failures,
       (int)active_qpus,
       (int)lane_width,
       GLOBAL_STORE_COHERENT_MAX_N,
       checksum_accum,
       (int)runtime_allocations,
       (int)runtime_launches,
       elapsed);

vc4_runtime_shutdown(&rt);

}

