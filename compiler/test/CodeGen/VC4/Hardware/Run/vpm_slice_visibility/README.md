# vpm_slice_visibility

## Purpose

`vpm_slice_visibility` is an exploratory VC4 QPU hardware-run test for the VPM
path.  It targets the storage-topology question from the slice/VPM visibility
plan:

- do all general-purpose QPUs see one shared first 4 KiB user-visible VPM
  window, or
- does each QPU slice see its own 4 KiB logical VPM window?

The test deliberately separates VPM storage visibility from VPM setup-state
sharing by taking the global QPU mutex around every VPM setup/access sequence in
these storage probes.

## Scope implemented by this test

This is the minimal first-run storage-visibility subset, not the entire future
matrix suite.  It implements:

1. a mandatory single-QPU VPM sanity check on QPU 0,
2. representative visibility pairs:
   - `(0, 1)` when `QUPS >= 2`,
   - `(0, QUPS)` and `(QUPS, 0)` when `NSLC >= 2`,
   - `(0, 2*QUPS)` and `(2*QUPS, 0)` when `NSLC >= 3`,
3. representative same-address collision pairs for `(0, 1)`, `(0, QUPS)`, and
   `(0, 2*QUPS)` where available, both write orders.

It does not yet run the full ordered-pair visibility matrix, the full unordered
collision matrix, the 64-row sweep/alias detector, or the no-mutex setup-clobber
matrix.  Those should be separate follow-up tests after this storage smoke test
passes on hardware.

## Hardware path exercised

The reference bundle uses general-purpose QPU user programs through the V3D
scheduler.  The QPU code reads `QPU_NUMBER` and `ELEMENT_NUMBER`, uses QPU
semaphores for pair ordering, uses the single global QPU mutex around VPM and
VDW setup/access, performs horizontal 32-bit VPM row writes/reads, and stores
16-lane result vectors to host memory through VDW.

The host launcher reads `V3D_IDENT1`, decodes `VPMSZ`, `QUPS`, and `NSLC`, writes
`V3D_VPMBASE = 16` to reserve the 4 KiB user-visible VPM window, and uses
`V3D_SQRSV0/1` to reserve all QPUs except the intended singleton or pair.

## QPU tag formats

The sanity vector is lane-distinctive:

```text
sanity[lane] = 0xD5000000 | lane
```

Visibility tags are:

```text
tag_visibility(writer, trial, lane) =
  0xA5000000 | ((trial & 15) << 20) | ((writer & 15) << 8) | lane
```

Collision tags are:

```text
tag_collision(writer, order, lane) =
  0xC5000000 | ((order & 15) << 20) | ((writer & 15) << 8) | lane
```

All 16 lanes must match for a vector match.

## Output

Bare-metal execution cannot create host filesystem files directly.  Instead the
harness prints file-like sections into `reference/run.log`:

```text
vc4_vpm_topology.json:
vc4_vpm_visibility_matrix.csv:
vc4_vpm_collision_matrix.csv:
vc4_vpm_full_log.txt:
```

The final semantic oracle line is:

```text
VC4_TEST_RESULT name=vpm_slice_visibility status=PASS ...
```

`expected.json` requires the hardware sanity check, same-slice visibility, and
same-slice collision to pass.  It requires no QPU-report mismatches, no request
timeouts, no invalid topology, and no relevant `V3D_ERRSTAT` bit changes.  It
does not require a particular cross-slice result; cross-slice observations are
printed and classified as `global_like`, `per_slice_like`, or
`mixed_or_inconclusive` through `conclusion_id`.

## Interpretation

A global user-visible VPM signature has cross-slice visibility matches and
cross-slice collision runs where both QPUs read the later writer's tag.

A per-slice VPM-window signature has cross-slice visibility misses and
cross-slice collision runs where each QPU reads the tag written in its own
slice.

The setup-clobber question is intentionally not answered here.  This test uses
the mutex to avoid conflating storage visibility with setup-state sharing.
