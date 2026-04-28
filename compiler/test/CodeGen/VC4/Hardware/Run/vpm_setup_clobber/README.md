
vpm_setup_clobber

vpm_setup_clobber is a hardware-run litmus test for VC4 generic VPM setup state sharing.

The test intentionally probes whether VPM generic read/write setup state is private per QPU, shared per slice, or shared globally. It is the right next incremental step after vpm_slice_visibility: that earlier test established that user-visible VPM storage behaves as one global 4 KiB window, but it deliberately protected setup and access with the global mutex and therefore did not answer whether another QPU can clobber VPM setup state between setup and access.

The hardware path exercised here is:

QPU reservations -> two selected physical QPUs
QPU semaphore ordering -> deterministic A/B setup window
VPM generic horizontal write setup -> intentionally unprotected clobber window
VPM_WRITE -> possible setup-state redirection
mutex-protected VPM readback -> VDW stores -> host result buffer

The key litmus sequence for each ordered physical pair (a, b) is:

A programs VPM write setup for CLOBBER_ROWA.
B programs VPM write setup for CLOBBER_ROWB.
A writes a lane-distinctive tag vector to VPM_WRITE without reprogramming setup.
A reads back both rows.

A row-A match indicates no setup clobber for that pair. A row-B match indicates B's setup redirected A's later write. Both/neither are reported as distinct classifications.

This test does not prove VPM storage visibility; that is already covered by vpm_slice_visibility. It does not test VDW setup clobbering independently. It does not measure performance and must not by itself justify changing runtime serialization policy unless the classifications remain stable over repeated supervised hardware runs.

Candidate-side codegen is intentionally disabled until the backend can emit this low-level diagnostic bundle.

Expected 12-QPU topology behavior

On the currently established Raspberry Pi VC4 topology (QUPS=4, NSLC=3, num_qpus=12), the harness runs these representative ordered pairs with three trials each:

(0, 1), (1, 0)
(0, 4), (4, 0)
(0, 8), (8, 0)

That gives representative_pairs=18, pairs_run=18, and valid_pairs=18 when the protocol runs cleanly.

The stable oracle intentionally does not require a specific setup-clobber classification. The classification is the hardware fact being discovered.

Files
input.mlir
expected.json
candidate/README.md
reference/.gitignore
reference/_harness.c
reference/vpm_setup_clobber.qasm
reference/vpm_setup_clobber_launch.c
reference/vpm_setup_clobber_launch.h

A mechanical Codex step copies/adapts the donor Makefile, run.sh, mailbox.c, mailbox.h, and share/ tree.

Do not add this test to compiler/test/CodeGen/VC4/catalog.json until a reference hardware run has passed and the result line has been validated against expected.json.

