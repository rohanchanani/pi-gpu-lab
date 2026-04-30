CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

OBJS = stencil2d_5point_shared_harness.o stencil2d_5point_shared_launch.o mailbox.o

all: stencil2d_5point_shared_harness

stencil2d_5point_shared_harness: $(OBJS)
$(CC) $(CFLAGS) $(OBJS) -o $@

%.o: %.c
$(CC) $(CFLAGS) -c $< -o $@

run: stencil2d_5point_shared_harness
./stencil2d_5point_shared_harness

clean:
rm -f $(OBJS) stencil2d_5point_shared_harness

.PHONY: all run clean
