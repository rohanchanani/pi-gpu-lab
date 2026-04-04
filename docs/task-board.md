---

### `docs/task-board.md`

```md
# Task board

## Now
- [ ] Add root-level repo guidance files
- [ ] Create compiler scaffold
- [ ] Set up MLIR project structure
- [ ] Add a minimal custom `vc4` dialect scaffold
- [ ] Define the first end-to-end milestone around SAXPY
- [x] Add a hand-written VC4 SAXPY kernel under `code/3-saxpy`

## Near term
- [ ] Represent a tiny scalar kernel form
- [ ] Lower scalar kernel form to width-16 vector form
- [ ] Add tests for vec16 lowering
- [ ] Lower vec16 form to a tiny `vc4` dialect subset
- [ ] Emit readable text or vc4asm-oriented output for SAXPY

## Later
- [ ] Decide whether the frontend path should remain toy/hardcoded or move toward parsing
- [ ] Add richer VC4 target ops as needed
- [ ] Add runtime launch path
- [ ] Add VPM-oriented abstractions
- [ ] Explore tiled kernels

## Constraints
- Keep v1 scope narrow
- Do not add `__shared__` in v1
- Do not add synchronization in v1
- Prefer architecture clarity over feature breadth

## Session notes
Use this section to leave short notes after major work:
- what changed
- what is blocked
- what the next best task is
- Added `code/3-saxpy` with a QPU SAXPY kernel, local runtime wrapper, and a CPU reference verifier.
- Local validation is limited here because `vc4asm` is not installed in this environment and the bare-metal target run requires Pi hardware.
- Next best task: run `code/3-saxpy/run.sh` on the Pi toolchain setup, then tune or fix the QPU kernel if the first hardware pass exposes VPM/DMA issues.
- Added `scripts/gen-compile-commands.sh` so CLion can index the real `make`-based code targets through a generated `compile_commands.json`.
