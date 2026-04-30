CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

.PHONY: run clean

run: gemv_naive_tail_harness_host
	./gemv_naive_tail_harness_host

gemv_naive_tail_harness_host: gemv_naive_tail_harness.c gemv_naive_tail_launch.c gemv_naive_tail_launch.h
	$(CC) $(CFLAGS) gemv_naive_tail_harness.c gemv_naive_tail_launch.c -o $@

clean:
	rm -f gemv_naive_tail_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
