CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic

.PHONY: run clean

run: image_boxblur_shared_harness_host
	./image_boxblur_shared_harness_host

image_boxblur_shared_harness_host: image_boxblur_shared_harness.c image_boxblur_shared_launch.c image_boxblur_shared_launch.h
	$(CC) $(CFLAGS) image_boxblur_shared_harness.c image_boxblur_shared_launch.c -o $@

clean:
	rm -f image_boxblur_shared_harness_host
	rm -rf objs
	rm -f *.bin *.elf *.list
