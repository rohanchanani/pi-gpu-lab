CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lm

.PHONY: run clean

run: flash_attention_tiny_harness_host
./flash_attention_tiny_harness_host

flash_attention_tiny_harness_host: flash_attention_tiny_harness.c flash_attention_tiny_launch.c flash_attention_tiny_launch.h
$(CC) $(CFLAGS) flash_attention_tiny_harness.c flash_attention_tiny_launch.c $(LDLIBS) -o $@

clean:
rm -f flash_attention_tiny_harness_host
rm -rf objs
rm -f *.bin *.elf *.list
