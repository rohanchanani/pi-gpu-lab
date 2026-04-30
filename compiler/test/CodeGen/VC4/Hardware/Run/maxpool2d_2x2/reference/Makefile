CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

.PHONY: run clean

run: maxpool2d_2x2_harness_host
	./maxpool2d_2x2_harness_host

maxpool2d_2x2_harness_host: maxpool2d_2x2_harness.c maxpool2d_2x2_launch.c maxpool2d_2x2_launch.h
	$(CC) $(CFLAGS) maxpool2d_2x2_harness.c maxpool2d_2x2_launch.c -o $@

clean:
	rm -f maxpool2d_2x2_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
