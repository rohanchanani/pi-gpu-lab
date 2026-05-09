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
static uint32_t vc4_stub_exec_count;

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

void mem_unlock(uint32_t handle) { (void)handle; }
void mem_free(uint32_t handle) { (void)handle; }

void gpu_fft_base_exec_direct(uint32_t code, uint32_t *unif_ptrs,
                              uint32_t num_qpus) {
  (void)code;
  (void)unif_ptrs;
  (void)num_qpus;
  vc4_stub_exec_count++;
}

uint32_t vc4_mailbox_stub_exec_count(void) { return vc4_stub_exec_count; }
uint32_t vc4_mailbox_stub_last_size(void) { return vc4_stub_last_size; }
uint32_t vc4_mailbox_stub_last_align(void) { return vc4_stub_last_align; }
uint32_t vc4_mailbox_stub_last_flags(void) { return vc4_stub_last_flags; }

void clean_dcache(void) {}
void flush_dcache(void) {}
void invalidate_dcache(void) {}
void delay_us(uint32_t usec) { (void)usec; }
void printk(const char *fmt, ...) { (void)fmt; }
void panic(const char *msg) { (void)msg; }
