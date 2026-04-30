CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: attention_qk_naive_harness_host
	./attention_qk_naive_harness_host

attention_qk_naive_harness_host: attention_qk_naive_harness.c attention_qk_naive_launch.c attention_qk_naive_launch.h
	$(CC) $(CFLAGS) attention_qk_naive_harness.c attention_qk_naive_launch.c $(LDLIBS) -o $@

clean:
	rm -f attention_qk_naive_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
