
shared_transpose_16x16

Purpose: validate a fixed 16x16 uint32_t transpose using VPM as CUDA-like shared memory, a reusable four-semaphore block barrier, vertical 32-bit VPM reads, and coalesced VDW output stores.

This is the right next incremental hardware-run test after vpm_slice_visibility, qpu_barrier_syncthreads, and matmul_blocked: those tests established global VPM visibility, reusable barriers, and useful VPM tiling. This test isolates the canonical shared-memory transpose orientation pattern: stage rows horizontally, synchronize, read columns vertically, and store transposed rows.

The hardware path exercised is:

TMU0 direct memory loads from the input tile.

Horizontal 32-bit VPM writes for row staging.

Four-semaphore reusable barrier with one resident 4-warp cooperative block.

Vertical 32-bit VPM reads for column vectors.

VPM staging plus VDW DMA stores for output rows.

Global mutex serialization around VPM/VDW setup and access.

This test does not prove arbitrary matrix sizes, byte/halfword transpose, arbitrary shared-memory scatter/gather, bank-conflict optimization, or VPM setup sharing without mutex protection.

Directory shape

The authored material files are:

compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/
  README.md
  input.mlir
  expected.json
  candidate/README.md
  reference/.gitignore
  reference/shared_transpose_16x16_harness.c
  reference/shared_transpose_16x16.qasm
  reference/shared_transpose_16x16_launch.c
  reference/shared_transpose_16x16_launch.h

The generated Codex mechanical prompt at repo root copies/adapts:

reference/Makefile
reference/run.sh
reference/mailbox.c
reference/mailbox.h
share/

Do not add this test to catalog.json until the reference hardware run passes and its run.log validates against expected.json.

Mechanical setup command

Run from repo root after the GPTWEB files have been parsed into the repository:

Bash
codex exec --dangerously-bypass-approvals-and-sandbox "$(cat _codex_mechanical_prompt.md)" && rm -f _codex_mechanical_prompt.md
chmod +x compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/run.sh
rm -rf compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/objs
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*.o
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*.d
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*.elf
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*.bin
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*.list
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/*shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/reference/run.log
Standard local checks
Bash
compiler/build/bin/vc4-opt \
  compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16/input.mlir \
  --vc4-verify-emit-contract \
  --vc4-verify-scheduled-hardware-rules \
  --vc4-verify-scheduled-adjacent-hazards \
  --vc4-verify-scheduled-io-spacing \
  --vc4-verify-scheduled-peripheral-accesses \
  -o /dev/null

cmake --build compiler/build --target check-vc4

compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/shared_transpose_16x16 \
  reference
