//===- heap_unit_test.c - Host-only VC4 generated heap API unit test -------===//
//
// This test mirrors the generated VC4 device heap contract on a host-only
// fixed buffer.  It exercises alignment, split, free, coalesce, invalid free,
// double free, OOM, HtoD/DtoH/DtoD copies, and D8 memset without requiring
// Raspberry Pi mailbox hardware.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint32_t vc4_deviceptr_t;

struct vc4_program;

int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
void vc4_program_destroy(struct vc4_program *program);
int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  const void *src, uint32_t bytes);
int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src,
                  uint32_t bytes);
int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  vc4_deviceptr_t src, uint32_t bytes);
int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst, uint8_t value,
                uint32_t bytes);

#define TEST_HEAP_BYTES 256u
#define TEST_GPU_BASE 0x10000000u
#define TEST_ALIGN 8u
#define TEST_NO_NEXT 0xffffffffu
#define TEST_MAGIC 0x48454150u

struct block_header {
  uint32_t size;
  uint32_t next;
  uint32_t free;
  uint32_t magic;
};

struct vc4_program {
  uint8_t heap[TEST_HEAP_BYTES];
  uint32_t heap_head;
  uint32_t heap_bytes;
  uint32_t allocs;
  uint32_t frees;
  uint32_t failures;
  uint32_t live_bytes;
  uint32_t high_water;
  uint32_t live;
};

static struct vc4_program g_program;

static uint32_t align_up(uint32_t value) {
  return (value + TEST_ALIGN - 1u) & ~(TEST_ALIGN - 1u);
}

static struct block_header *block_at(struct vc4_program *program,
                                     uint32_t offset) {
  if (!program || !program->live)
    return 0;
  if (offset > program->heap_bytes ||
      program->heap_bytes - offset < sizeof(struct block_header))
    return 0;
  struct block_header *block =
      (struct block_header *)(void *)(program->heap + offset);
  if (block->magic != TEST_MAGIC)
    return 0;
  return block;
}

static void fail_heap(struct vc4_program *program) {
  if (program && program->live)
    program->failures++;
}

static void init_heap(struct vc4_program *program) {
  memset(program, 0, sizeof(*program));
  program->live = 1u;
  program->heap_bytes = TEST_HEAP_BYTES;
  program->heap_head = 0u;
  struct block_header *head = (struct block_header *)(void *)program->heap;
  head->size = TEST_HEAP_BYTES - (uint32_t)sizeof(struct block_header);
  head->next = TEST_NO_NEXT;
  head->free = 1u;
  head->magic = TEST_MAGIC;
}

static void coalesce_next(struct vc4_program *program,
                          struct block_header *block) {
  while (block->next != TEST_NO_NEXT) {
    struct block_header *next = block_at(program, block->next);
    if (!next || !next->free)
      return;
    block->size += (uint32_t)sizeof(struct block_header) + next->size;
    block->next = next->next;
  }
}

static int range_offset(struct vc4_program *program, vc4_deviceptr_t ptr,
                        uint32_t bytes, uint32_t *offset_out) {
  if (!program || !program->live || ptr < TEST_GPU_BASE)
    return 0;
  uint32_t offset = ptr - TEST_GPU_BASE;
  if (offset > program->heap_bytes || bytes > program->heap_bytes - offset)
    return 0;
  if (offset_out)
    *offset_out = offset;
  return 1;
}

static int range_is_allocated(struct vc4_program *program, vc4_deviceptr_t ptr,
                              uint32_t bytes) {
  uint32_t offset = 0u;
  if (!range_offset(program, ptr, bytes, &offset))
    return 0;
  for (uint32_t current = program->heap_head; current != TEST_NO_NEXT;) {
    struct block_header *block = block_at(program, current);
    if (!block)
      return 0;
    uint32_t user_begin = current + (uint32_t)sizeof(struct block_header);
    if (!block->free && offset >= user_begin &&
        offset - user_begin <= block->size &&
        bytes <= block->size - (offset - user_begin))
      return 1;
    current = block->next;
  }
  return 0;
}

static void *ptr_to_host(struct vc4_program *program, vc4_deviceptr_t ptr,
                         uint32_t bytes) {
  if (!range_is_allocated(program, ptr, bytes))
    return 0;
  return program->heap + (ptr - TEST_GPU_BASE);
}

int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes) {
  if (!out || requested_bytes > TEST_HEAP_BYTES - sizeof(struct block_header))
    return -1;
  init_heap(&g_program);
  *out = &g_program;
  return 0;
}

void vc4_program_destroy(struct vc4_program *program) {
  if (program)
    program->live = 0u;
}

int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out,
              uint32_t bytes) {
  if (!program || !program->live || !out) {
    fail_heap(program);
    return -1;
  }
  *out = 0u;
  if (bytes == 0u)
    return 0;
  uint32_t need = align_up(bytes);
  for (uint32_t current = program->heap_head; current != TEST_NO_NEXT;) {
    struct block_header *block = block_at(program, current);
    if (!block)
      break;
    if (block->free && block->size >= need) {
      uint32_t remaining = block->size - need;
      if (remaining > sizeof(struct block_header) + TEST_ALIGN) {
        uint32_t split_offset = current + sizeof(struct block_header) + need;
        struct block_header *split =
            (struct block_header *)(void *)(program->heap + split_offset);
        split->size = remaining - (uint32_t)sizeof(struct block_header);
        split->next = block->next;
        split->free = 1u;
        split->magic = TEST_MAGIC;
        block->size = need;
        block->next = split_offset;
      }
      block->free = 0u;
      program->allocs++;
      program->live_bytes += block->size;
      if (program->live_bytes > program->high_water)
        program->high_water = program->live_bytes;
      *out = TEST_GPU_BASE + current + (uint32_t)sizeof(struct block_header);
      return 0;
    }
    current = block->next;
  }
  fail_heap(program);
  return -1;
}

int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr) {
  if (!program || !program->live) {
    fail_heap(program);
    return -1;
  }
  if (ptr == 0u)
    return 0;
  uint32_t offset = 0u;
  if (!range_offset(program, ptr, 0u, &offset) ||
      offset < sizeof(struct block_header)) {
    fail_heap(program);
    return -1;
  }
  uint32_t block_offset = offset - (uint32_t)sizeof(struct block_header);
  uint32_t prev = TEST_NO_NEXT;
  for (uint32_t current = program->heap_head; current != TEST_NO_NEXT;) {
    struct block_header *block = block_at(program, current);
    if (!block)
      break;
    if (current == block_offset) {
      if (block->free) {
        fail_heap(program);
        return -1;
      }
      block->free = 1u;
      program->frees++;
      program->live_bytes = program->live_bytes >= block->size
                                ? program->live_bytes - block->size
                                : 0u;
      coalesce_next(program, block);
      if (prev != TEST_NO_NEXT) {
        struct block_header *prev_block = block_at(program, prev);
        if (prev_block && prev_block->free)
          coalesce_next(program, prev_block);
      }
      return 0;
    }
    prev = current;
    current = block->next;
  }
  fail_heap(program);
  return -1;
}

int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  const void *src, uint32_t bytes) {
  if (bytes == 0u)
    return 0;
  void *host = ptr_to_host(program, dst, bytes);
  if (!host || !src) {
    fail_heap(program);
    return -1;
  }
  memcpy(host, src, bytes);
  return 0;
}

int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src,
                  uint32_t bytes) {
  if (bytes == 0u)
    return 0;
  void *host = ptr_to_host(program, src, bytes);
  if (!host || !dst) {
    fail_heap(program);
    return -1;
  }
  memcpy(dst, host, bytes);
  return 0;
}

int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst,
                  vc4_deviceptr_t src, uint32_t bytes) {
  if (bytes == 0u)
    return 0;
  void *dst_host = ptr_to_host(program, dst, bytes);
  void *src_host = ptr_to_host(program, src, bytes);
  if (!dst_host || !src_host) {
    fail_heap(program);
    return -1;
  }
  memmove(dst_host, src_host, bytes);
  return 0;
}

int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst,
                uint8_t value, uint32_t bytes) {
  if (bytes == 0u)
    return 0;
  void *host = ptr_to_host(program, dst, bytes);
  if (!host) {
    fail_heap(program);
    return -1;
  }
  memset(host, value, bytes);
  return 0;
}

static int expect(int cond, const char *label) {
  if (cond)
    return 0;
  printf("HEAP_UNIT_FAIL label=%s\n", label);
  return 1;
}

int main(void) {
  struct vc4_program *program = 0;
  uint32_t failures = 0;
  vc4_deviceptr_t a = 0, b = 0, c = 0, d = 0;
  uint8_t host_in[24];
  uint8_t host_out[24];
  for (uint32_t i = 0; i != sizeof(host_in); ++i)
    host_in[i] = (uint8_t)(0xa0u + i);

  failures += expect(vc4_program_create(&program, 0) == 0 && program,
                     "program_create");
  failures += expect(vc4Malloc(program, &a, 13) == 0 && a % TEST_ALIGN == 0,
                     "alloc_a_align");
  failures += expect(vc4Malloc(program, &b, 24) == 0 && b != 0 && b != a,
                     "alloc_b");
  failures += expect(vc4MemcpyHtoD(program, a, host_in, 13) == 0,
                     "copy_htod");
  memset(host_out, 0, sizeof(host_out));
  failures += expect(vc4MemcpyDtoH(program, host_out, a, 13) == 0 &&
                         memcmp(host_in, host_out, 13) == 0,
                     "copy_dtoh");
  failures += expect(vc4MemsetD8(program, b, 0x5a, 24) == 0, "memset_d8");
  failures += expect(vc4MemcpyDtoD(program, a, b, 13) == 0, "copy_dtod");
  memset(host_out, 0, sizeof(host_out));
  failures += expect(vc4MemcpyDtoH(program, host_out, a, 13) == 0,
                     "copy_dtod_readback");
  for (uint32_t i = 0; i != 13; ++i)
    failures += expect(host_out[i] == 0x5a, "copy_dtod_value");
  failures += expect(vc4Free(program, a) == 0, "free_a");
  failures += expect(vc4Free(program, a) < 0, "double_free_a");
  failures += expect(vc4Free(program, TEST_GPU_BASE + TEST_HEAP_BYTES + 16u) < 0,
                     "invalid_free");
  failures += expect(vc4Malloc(program, &c, 16) == 0 && c == a,
                     "reuse_split_block");
  failures += expect(vc4Free(program, c) == 0, "free_c");
  failures += expect(vc4Free(program, b) == 0, "free_b_coalesce");
  failures += expect(vc4Malloc(program, &d, 192) == 0,
                     "large_alloc_after_coalesce");
  failures += expect(vc4Free(program, d) == 0, "free_large");
  failures += expect(vc4Malloc(program, &d, TEST_HEAP_BYTES) < 0, "oom");
  failures += expect(program->allocs >= 4 && program->frees >= 4 &&
                         program->failures >= 3 && program->high_water >= 24,
                     "stats");
  vc4_program_destroy(program);

  if (failures == 0)
    printf("HEAP_UNIT_RESULT status=PASS allocs=%u frees=%u failures=%u high_water=%u\n",
           g_program.allocs, g_program.frees, g_program.failures,
           g_program.high_water);
  else
    printf("HEAP_UNIT_RESULT status=FAIL failures=%u\n", failures);
  return failures == 0 ? 0 : 1;
}
