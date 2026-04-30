CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

.PHONY: run clean

run: image_sobel_3x3_harness_host
	./image_sobel_3x3_harness_host

image_sobel_3x3_harness_host: image_sobel_3x3_harness.c image_sobel_3x3_launch.c image_sobel_3x3_launch.h
	$(CC) $(CFLAGS) image_sobel_3x3_harness.c image_sobel_3x3_launch.c -o $@

clean:
	rm -f image_sobel_3x3_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
