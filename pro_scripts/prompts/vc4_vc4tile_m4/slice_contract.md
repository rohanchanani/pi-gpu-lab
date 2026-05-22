# VC4 VC4Tile M4 Slice Contract

Each slice must preserve the milestone path:

```text
vc4tile input
  -> vc4-opt --convert-vc4tile-to-ssavc4
  -> ssavc4
  -> vc4-opt --convert-ssavc4-to-vc4
  -> scheduled vc4
  -> existing scheduled verifiers
  -> vc4-codegen --emit-bundle
  -> existing artifact/runtime path
```

Feature implementation rule:

```text
Every implemented vc4tile feature must pass:
  dialect contract
  invalid diagnostic contract
  vc4tile -> ssavc4 lowered-IR contract
  ssavc4 -> vc4 scheduled/artifact contract
  hardware CPU-reference contract when executable on hardware
```

Hardware is the gold standard. If the feature can be executed on VC4 hardware, it should have a hardware fixture and reference comparison. If a feature is metadata-only or intentionally not yet executable, the feature gate must document why hardware is not required for that feature in that slice.

No slice may make future-slice tests active before the owning implementation exists. Global `check-vc4` must remain green.
