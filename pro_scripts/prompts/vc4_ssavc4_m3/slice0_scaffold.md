# m3-00-milestone-package: Milestone package smoke

    Goal: validate the M3 milestone package itself. Do not implement compiler source changes in this slice.

Required work:
- Ensure the descriptor, worklist, verification spec, context profiles, prompt handoffs, and plan document exist at the exact repo-relative paths in the user contract.
- Ensure all JSON files parse and the M3 spec audits against the M3 worklist.
- Keep `m3-00-milestone-package` free of build or hardware requirements so it passes immediately after package placement.

Forbidden work:
- Do not add `compiler/include/vc4/Dialect/SSAVC4/**` yet.
- Do not edit generic milestone drivers.
- Do not add implementation fixtures, support scripts, or compiler code.

Verification:
```bash
python3 pro_scripts/vc4_milestone_verifier.py verify \
  --repo "$PWD" \
  --milestone-config pro_scripts/milestones/vc4-ssavc4-m3.json \
  --slice m3-00-milestone-package \
  --out /tmp/m3-00-package-smoke.json \
  --timeout-sec 7200 \
  --keep-going
```

    ## Report at the end of the slice

    Report the changed source files, the exact verifier command(s) run, whether hardware was required, and the first failing verifier packet if anything remains red. Do not include private reasoning. Do not claim success until the typed verifier entry for `m3-00-milestone-package` passes.
