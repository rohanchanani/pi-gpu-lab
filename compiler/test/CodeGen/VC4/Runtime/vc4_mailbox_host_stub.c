#include <stdint.h>

/* Host-only placeholder for runtime unit tests. Hardware mailbox functions are
 * supplied by libpi in real Pi builds; host tests should not touch them. */
uint32_t vc4_mailbox_host_stub_anchor(void) { return 0u; }
