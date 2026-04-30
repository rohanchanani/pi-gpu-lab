CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: softmax_row_harness_host
./softmax_row_harness_host

softmax_row_harness_host: softmax_row_harness.c softmax_row_launch.c softmax_row_launch.h
$(CC) $(CFLAGS) softmax_row_harness.c softmax_row_launch.c $(LDLIBS) -o $@

clean:
rm -f softmax_row_harness_host
rm -rf objs
rm -f *.bin *.elf *.list
