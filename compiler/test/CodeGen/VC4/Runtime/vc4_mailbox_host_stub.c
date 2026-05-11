//===- vc4_mailbox_host_stub.c - Host stubs for VC4 runtime unit tests ----===//
//
// Host-only tests use this file to satisfy generated-runtime mailbox symbols
// when a unit links against generated launcher C.  No real mailbox or QPU
// hardware access is performed.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

static uint8_t vc4_stub_memory[65536];
static uint32_t vc4_stub_next_handle = 1u;
static uint32_t vc4_stub_last_size;
static uint32_t vc4_stub_last_align;
static uint32_t vc4_stub_last_flags;
static uint32_t vc4_stub_srqpc_writes;
static uint32_t vc4_stub_srqua_writes;
static uint32_t vc4_stub_completed;
static uint32_t vc4_stub_complete_on_launch = 1u;
static uint32_t vc4_stub_now_usec;
static uint32_t vc4_stub_time_step_usec;

void vc4_mailbox_stub_reset(void) {
  memset(vc4_stub_memory, 0, sizeof(vc4_stub_memory));
  vc4_stub_last_size = 0u;
  vc4_stub_last_align = 0u;
  vc4_stub_last_flags = 0u;
  vc4_stub_srqpc_writes = 0u;
  vc4_stub_srqua_writes = 0u;
  vc4_stub_completed = 0u;
  vc4_stub_complete_on_launch = 1u;
  vc4_stub_now_usec = 0u;
  vc4_stub_time_step_usec = 0u;
}

uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags) {
  vc4_stub_last_size = size;
  vc4_stub_last_align = align;
  vc4_stub_last_flags = flags;
  if (size == 0u || size > sizeof(vc4_stub_memory))
    return 0u;
  return vc4_stub_next_handle++;
}

uint32_t mem_lock(uint32_t handle) {
  if (!handle)
    return 0u;
  return 0x40000000u + (uint32_t)(uintptr_t)&vc4_stub_memory[0];
}

uint32_t mem_unlock(uint32_t handle) {
  (void)handle;
  return 0u;
}

uint32_t mem_free(uint32_t handle) {
  (void)handle;
  return 0u;
}

uint32_t vc4_runtime_test_get32(uint32_t addr) {
  if (addr == 0x20c0043cu)
    return vc4_stub_completed << 16;
  return 0u;
}

void vc4_runtime_test_put32(uint32_t addr, uint32_t value) {
  (void)value;
  if (addr == 0x20c0043cu) {
    vc4_stub_completed = 0u;
    return;
  }
  if (addr == 0x20c00434u) {
    vc4_stub_srqua_writes++;
    return;
  }
  if (addr == 0x20c00430u) {
    vc4_stub_srqpc_writes++;
    if (vc4_stub_complete_on_launch)
      vc4_stub_completed++;
  }
}

uint32_t vc4_runtime_test_now_usec(void) {
  uint32_t now = vc4_stub_now_usec;
  vc4_stub_now_usec += vc4_stub_time_step_usec;
  return now;
}

void *vc4_runtime_test_host_ptr(uint32_t bus_addr) {
  uint32_t base = 0x40000000u + (uint32_t)(uintptr_t)&vc4_stub_memory[0];
  if (bus_addr < base)
    return 0;
  uint32_t offset = bus_addr - base;
  if (offset >= sizeof(vc4_stub_memory))
    return 0;
  return &vc4_stub_memory[offset];
}

void vc4_mailbox_stub_set_complete_on_launch(uint32_t enabled) {
  vc4_stub_complete_on_launch = enabled ? 1u : 0u;
}

void vc4_mailbox_stub_set_time_step_usec(uint32_t usec) {
  vc4_stub_time_step_usec = usec;
}

uint32_t vc4_mailbox_stub_srqpc_writes(void) { return vc4_stub_srqpc_writes; }
uint32_t vc4_mailbox_stub_srqua_writes(void) { return vc4_stub_srqua_writes; }
uint32_t vc4_mailbox_stub_last_size(void) { return vc4_stub_last_size; }
uint32_t vc4_mailbox_stub_last_align(void) { return vc4_stub_last_align; }
uint32_t vc4_mailbox_stub_last_flags(void) { return vc4_stub_last_flags; }

void clean_dcache(void) {}
void flush_dcache(void) {}
void invalidate_dcache(void) {}
void delay_us(uint32_t usec) { (void)usec; }
void printk(const char *fmt, ...) { (void)fmt; }
void panic(const char *msg) { (void)msg; }
