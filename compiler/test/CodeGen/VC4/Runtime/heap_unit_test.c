#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct Block {
    uint32_t off;
    uint32_t size;
    uint8_t free;
    struct Block *next;
} Block;

#define HEAP_BYTES 4096u
#define MAX_BLOCKS 64u
static uint8_t heap[HEAP_BYTES];
static Block blocks[MAX_BLOCKS];
static uint32_t block_count;

static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1u) & ~(a - 1u); }

static void heap_init(void) {
    memset(heap, 0, sizeof(heap));
    memset(blocks, 0, sizeof(blocks));
    block_count = 1;
    blocks[0].off = 0;
    blocks[0].size = HEAP_BYTES;
    blocks[0].free = 1;
    blocks[0].next = 0;
}

static Block *new_block(uint32_t off, uint32_t size, uint8_t free) {
    if (block_count >= MAX_BLOCKS) return 0;
    Block *b = &blocks[block_count++];
    b->off = off; b->size = size; b->free = free; b->next = 0;
    return b;
}

static int heap_alloc(uint32_t bytes, uint32_t *ptr) {
    bytes = align_up(bytes, 8);
    for (Block *b = &blocks[0]; b; b = b->next) {
        if (!b->free || b->size < bytes) continue;
        if (b->size >= bytes + 8) {
            Block *rest = new_block(b->off + bytes, b->size - bytes, 1);
            if (!rest) return -1;
            rest->next = b->next;
            b->next = rest;
            b->size = bytes;
        }
        b->free = 0;
        *ptr = b->off;
        return 0;
    }
    return -1;
}

static int heap_free(uint32_t ptr) {
    Block *prev = 0;
    for (Block *b = &blocks[0]; b; prev = b, b = b->next) {
        if (b->off != ptr) continue;
        if (b->free) return -1;
        b->free = 1;
        if (b->next && b->next->free) {
            b->size += b->next->size;
            b->next = b->next->next;
        }
        if (prev && prev->free) {
            prev->size += b->size;
            prev->next = b->next;
        }
        return 0;
    }
    return -1;
}

static int range_ok(uint32_t ptr, uint32_t bytes) { return ptr <= HEAP_BYTES && bytes <= HEAP_BYTES - ptr; }

int main(void) {
    uint32_t a=0, b=0, c=0, d=0;
    int alignment_pass = 1, coalesce_pass = 0, invalid_free_rejected = 0, oom_pass = 0;
    heap_init();
    if (heap_alloc(13, &a) || heap_alloc(24, &b) || heap_alloc(7, &c)) return 2;
    alignment_pass = ((a|b|c) & 7u) == 0u;
    if (!range_ok(a, 13) || !range_ok(b, 24) || !range_ok(c, 7)) return 3;
    invalid_free_rejected = heap_free(HEAP_BYTES + 8u) < 0 && heap_free(b) == 0 && heap_free(b) < 0;
    if (heap_free(a) || heap_free(c)) return 4;
    coalesce_pass = heap_alloc(HEAP_BYTES, &d) == 0 && d == 0;
    oom_pass = heap_alloc(8, &a) < 0;
    int ok = alignment_pass && coalesce_pass && invalid_free_rejected && oom_pass;
    printf("HEAP_UNIT_RESULT status=%s coalesce_pass=%d alignment_pass=%d invalid_free_rejected=%d oom_pass=%d\n",
           ok ? "PASS" : "FAIL", coalesce_pass, alignment_pass, invalid_free_rejected, oom_pass);
    return ok ? 0 : 1;
}
