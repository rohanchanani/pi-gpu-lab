CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: mandelbrot_masked_loop_harness_host
./mandelbrot_masked_loop_harness_host

mandelbrot_masked_loop_harness_host: mandelbrot_masked_loop_harness.c mandelbrot_masked_loop_launch.c mandelbrot_masked_loop_launch.h
$(CC) $(CFLAGS) mandelbrot_masked_loop_harness.c mandelbrot_masked_loop_launch.c $(LDLIBS) -o $@

clean:
rm -f mandelbrot_masked_loop_harness_host
rm -rf objs
rm -f *.bin *.elf *.list
