CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: layernorm_row_harness_host
	./layernorm_row_harness_host

layernorm_row_harness_host: layernorm_row_harness.c layernorm_row_launch.c layernorm_row_launch.h
	$(CC) $(CFLAGS) layernorm_row_harness.c layernorm_row_launch.c $(LDLIBS) -o $@

clean:
	rm -f layernorm_row_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
