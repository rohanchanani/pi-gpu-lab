# VC4 SSAVC4 M3 Prompt Package

This directory contains the M3 prompt and contract material for `vc4-ssavc4-m3`.
M3 is the milestone that defines the new target-specific `ssavc4` dialect and implements the incremental lowering from `ssavc4` to the existing scheduled `vc4` sink.

The package intentionally does not contain compiler implementation files. In particular, it does not create `SSAVC4Ops.td`, conversion-pass source, fixtures, or support scripts. Those are slice products driven by the worklist and verification spec.

## Files in this prompt directory

- `constitution.md` — non-negotiable project rules for all M3 attempts.
- `output_contract.md` — downloadable-bundle response contract for implementation attempts.
- `slice_contract.md` — cross-slice implementation and verification contract.
- `m3_handoff.md` — compact milestone handoff derived from the post-cleanup SSAVC4 design and current M2 state.
- `slice0_scaffold.md` through `slice10_final_acceptance.md` — concrete per-slice handoffs.

The generic milestone entry point is:

```bash
python3 pro_scripts/vc4_milestone_resume.py   --repo "$PWD"   --milestone-config pro_scripts/milestones/vc4-ssavc4-m3.json   --gpt-mode current_tab   --timeout-sec 7200
```

Run the m3-00 package smoke immediately after placement before starting implementation.

The detailed post-cleanup design document is installed at `compiler/docs/codegen/ssavc4-ir-design-m3-post-cleanup.md`.
