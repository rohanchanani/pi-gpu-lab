//===- raw_srq_unit_test.c - Host-only VC4 raw SRQ runtime tests ----------===//
//
// This test links the invariant libpi vc4_runtime.c with VC4_RUNTIME_TESTING
// enabled.  The mailbox host stub supplies deterministic V3D register and time
// hooks, so the test exercises vc4LaunchKernel queueing without QPU hardware.
//
//===----------------------------------------------------------------------===//

#include "vc4_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

void vc4_mailbox_stub_reset(void);
void vc4_mailbox_stub_set_complete_on_launch(uint32_t enabled);
void vc4_mailbox_stub_set_time_step_usec(uint32_t usec);
uint32_t vc4_mailbox_stub_srqpc_writes(void);
uint32_t vc4_mailbox_stub_srqua_writes(void);
void vc4_runtime_test_set_active_qpus(struct vc4_program *program,
                                      uint32_t active_qpus);

static const uint32_t test_shader[] = {
    0x009e7000u,
    0x009e7000u,
    0x009e7000u,
};

static const struct vc4_kernel_image test_kernels[] = {
    {"raw_srq_test", test_shader,
     (uint32_t)(sizeof(test_shader) / sizeof(test_shader[0])), 3u,
     VC4_RUNTIME_MAX_QPUS, VC4_KERNEL_SCHEDULE_INDEPENDENT_VECTOR, 0u},
};

static const struct vc4_module_image test_module = {
    VC4_RUNTIME_MODULE_VERSION,
    (uint32_t)(sizeof(test_kernels) / sizeof(test_kernels[0])),
    4096u,
    test_kernels,
};

struct pack_state {
  uint32_t calls;
  uint32_t fail_at;
  uint32_t saw_bad_words;
  uint32_t logical_sum;
};

static int pack_uniforms(void *opaque, uint32_t logical_request,
                         uint32_t *uniform_words,
                         uint32_t uniform_words_per_request) {
  struct pack_state *state = (struct pack_state *)opaque;
  if (!state || !uniform_words || uniform_words_per_request != 3u)
    return -1;
  if (state->fail_at != 0xffffffffu && logical_request == state->fail_at)
    return -1;
  uniform_words[0] = 0xabc00000u | logical_request;
  uniform_words[1] = logical_request;
  uniform_words[2] = uniform_words_per_request;
  state->calls++;
  state->logical_sum += logical_request;
  if (uniform_words_per_request != 3u)
    state->saw_bad_words = 1u;
  return 0;
}

static int create_program(struct vc4_program **program) {
  vc4_mailbox_stub_reset();
  return vc4ProgramCreateFromImage(program, &test_module, 4096u);
}

static int expect_u32(const char *label, uint32_t actual, uint32_t expected) {
  if (actual == expected)
    return 1;
  printf("RAW_SRQ_CHECK_FAIL label=%s actual=%u expected=%u\n", label, actual,
         expected);
  return 0;
}

static int test_zero_requests(void) {
  struct vc4_program *program = 0;
  if (create_program(&program) < 0)
    return 0;
  int ok = vc4LaunchKernel(program, 0u, 0u, 0, 0) == 0;
  ok &= expect_u32("zero_launches", vc4ProgramLaunches(program), 1u);
  ok &= expect_u32("zero_failures", vc4ProgramLaunchFailures(program), 0u);
  ok &= expect_u32("zero_srqpc", vc4_mailbox_stub_srqpc_writes(), 0u);
  vc4_program_destroy(program);
  return ok;
}

static int test_one_request(void) {
  struct vc4_program *program = 0;
  struct pack_state state = {0u, 0xffffffffu, 0u, 0u};
  if (create_program(&program) < 0)
    return 0;
  int ok = vc4LaunchKernel(program, 0u, 1u, pack_uniforms, &state) == 0;
  ok &= expect_u32("one_pack_calls", state.calls, 1u);
  ok &= expect_u32("one_srqpc", vc4_mailbox_stub_srqpc_writes(), 1u);
  ok &= expect_u32("one_srqua", vc4_mailbox_stub_srqua_writes(), 1u);
  ok &= expect_u32("one_launches", vc4ProgramLaunches(program), 1u);
  ok &= expect_u32("one_failures", vc4ProgramLaunchFailures(program), 0u);
  vc4_program_destroy(program);
  return ok;
}

static int test_two_waves(void) {
  struct vc4_program *program = 0;
  struct pack_state state = {0u, 0xffffffffu, 0u, 0u};
  if (create_program(&program) < 0)
    return 0;
  uint32_t requests = VC4_RUNTIME_MAX_QPUS + 1u;
  int ok = vc4LaunchKernel(program, 0u, requests, pack_uniforms, &state) == 0;
  ok &= expect_u32("two_waves_pack_calls", state.calls, requests);
  ok &= expect_u32("two_waves_srqpc", vc4_mailbox_stub_srqpc_writes(),
                   requests);
  ok &= expect_u32("two_waves_srqua", vc4_mailbox_stub_srqua_writes(),
                   requests);
  ok &= expect_u32("two_waves_launches", vc4ProgramLaunches(program), 1u);
  ok &= expect_u32("two_waves_failures", vc4ProgramLaunchFailures(program), 0u);
  vc4_program_destroy(program);
  return ok;
}

static int test_pack_failure(void) {
  struct vc4_program *program = 0;
  struct pack_state state = {0u, 1u, 0u, 0u};
  if (create_program(&program) < 0)
    return 0;
  int ok = vc4LaunchKernel(program, 0u, 2u, pack_uniforms, &state) < 0;
  ok &= expect_u32("pack_failure_calls", state.calls, 1u);
  ok &= expect_u32("pack_failure_srqpc", vc4_mailbox_stub_srqpc_writes(), 0u);
  ok &= expect_u32("pack_failure_launches", vc4ProgramLaunches(program), 0u);
  ok &= expect_u32("pack_failure_failures", vc4ProgramLaunchFailures(program),
                   1u);
  vc4_program_destroy(program);
  return ok;
}

static int test_timeout(void) {
  struct vc4_program *program = 0;
  struct pack_state state = {0u, 0xffffffffu, 0u, 0u};
  if (create_program(&program) < 0)
    return 0;
  vc4_mailbox_stub_set_complete_on_launch(0u);
  vc4_mailbox_stub_set_time_step_usec(2000001u);
  int ok = vc4LaunchKernel(program, 0u, 1u, pack_uniforms, &state) < 0;
  ok &= expect_u32("timeout_pack_calls", state.calls, 1u);
  ok &= expect_u32("timeout_srqpc", vc4_mailbox_stub_srqpc_writes(), 1u);
  ok &= expect_u32("timeout_launches", vc4ProgramLaunches(program), 0u);
  ok &= expect_u32("timeout_failures", vc4ProgramLaunchFailures(program), 1u);
  vc4_program_destroy(program);
  return ok;
}

static int test_active_qpu_bounds(void) {
  struct vc4_program *program = 0;
  struct pack_state state = {0u, 0xffffffffu, 0u, 0u};
  if (create_program(&program) < 0)
    return 0;
  vc4_runtime_test_set_active_qpus(program, 0u);
  int ok = vc4LaunchKernel(program, 0u, 1u, pack_uniforms, &state) < 0;
  ok &= expect_u32("active_zero_failures", vc4ProgramLaunchFailures(program),
                   1u);
  vc4_runtime_test_set_active_qpus(program, VC4_RUNTIME_MAX_QPUS + 1u);
  ok &= vc4LaunchKernel(program, 0u, 1u, pack_uniforms, &state) < 0;
  ok &= expect_u32("active_too_many_failures",
                   vc4ProgramLaunchFailures(program), 2u);
  ok &= expect_u32("active_bounds_launches", vc4ProgramLaunches(program), 0u);
  ok &= expect_u32("active_bounds_srqpc", vc4_mailbox_stub_srqpc_writes(), 0u);
  vc4_program_destroy(program);
  return ok;
}

int main(void) {
  int ok = 1;
  ok &= test_zero_requests();
  ok &= test_one_request();
  ok &= test_two_waves();
  ok &= test_pack_failure();
  ok &= test_timeout();
  ok &= test_active_qpu_bounds();
  printf("VC4_RAW_SRQ_UNIT_RESULT status=%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
