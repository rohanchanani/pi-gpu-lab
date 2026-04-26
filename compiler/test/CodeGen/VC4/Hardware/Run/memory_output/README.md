# VC4 hardware contract test: memory_output

This is the second runnable hardware contract test in the VC4 codegen corpus.
It proves that a trusted reference bundle can launch a QPU user program that
writes observable data back to host-visible memory.

## What this test verifies

- The bare-metal VC4 runtime can enable QPUs and submit general-purpose QPU
  user-program requests.
- The reference qasm can write one 16-lane vector per active QPU through
  VPM/VDW into a host-visible output buffer.
- Per-QPU uniform packing is hidden inside the launcher.
- The public launcher API exposes only semantic arguments plus the runtime
  handle.
- The final result is reported through a stable `VC4_TEST_RESULT` line.

The kernel writes one row per active QPU.  For QPU `q`, all 16 output words
in that row are `q`.

With the current reference runtime policy of 12 active QPUs, the expected
output is:

```text
output[0..15]     = 0
output[16..31]    = 1
...
output[176..191]  = 11
```

The expected checksum is:

```text
sum(q * 16 for q in 0..11) = 1056
```

## Contract files

- `input.mlir` is the final-stage `vc4` MLIR input intended for future
  candidate codegen.
- `expected.json` is the machine-readable semantic oracle.
- `reference/` contains the trusted hand-authored qasm/C/H bundle and
  bare-metal harness.
- `candidate/` is a placeholder for future generated-code runs.

## Running

From repo root:

```bash
compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/memory_output \
  reference
```

The expected result line is:

```text
VC4_TEST_RESULT name=memory_output status=PASS mismatches=0 active_qpus=12 words=192 checksum=1056 ...
```

## Notes

This test deliberately uses VPM/VDW, so it is more hardware-facing than
`minimal_thrend` but still avoids input DMA, TMU, SFU, branches, and tails.
