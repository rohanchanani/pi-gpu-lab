CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: attention_naive_tiny_harness_host
	./attention_naive_tiny_harness_host

attention_naive_tiny_harness_host: attention_naive_tiny_harness.c attention_naive_tiny_launch.c attention_naive_tiny_launch.h
	$(CC) $(CFLAGS) attention_naive_tiny_harness.c attention_naive_tiny_launch.c $(LDLIBS) -o $@

clean:
	rm -f attention_naive_tiny_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
