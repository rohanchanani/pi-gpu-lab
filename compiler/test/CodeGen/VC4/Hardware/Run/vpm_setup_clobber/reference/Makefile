LIBS += $(CS240LX_2025_PATH)/lib/libgcc.a $(CS240LX_2025_PATH)/libpi/libpi.a

export OPT_LEVEL := -O3

COMMON_SRC := mailbox.c vpm_setup_clobber_launch.c shader.c

PROGS := vpm_setup_clobber_harness.c

STAFF_OBJS += $(CS240LX_2025_PATH)/libpi/staff-objs/staff-hw-spi.o
STAFF_OBJS += $(CS240LX_2025_PATH)/libpi/staff-objs/kmalloc.o

RUN ?= 1

BOOTLOADER = pi-install
EXCLUDE ?= grep -v simple_boot
GREP_STR := 'HASH:|ERROR:|PANIC:|SUCCESS:|NRF:'
include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust
