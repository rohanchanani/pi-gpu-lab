# VC4 VC4Tile M5 Slice Contract

Each implemented M5 feature must pass the applicable layers: source products, dialect/type/attr contract, invalid diagnostics, surface-to-core contract, copy-plan contract when movement is involved, precision policy contract when precision fields are accepted, lowered-IR contract, scheduled/artifact contract when executable, hardware CPU-reference contract when executable, and implementation-integrity scans.

Hardware fixtures must be candidate-first and fresh by default. Harnesses must call generated launch wrappers, copy device outputs back, compare against CPU oracles, check sentinels, and derive `VC4_TEST_RESULT` from real mismatches and launch failures.

Failure attempts are focused repairs. Preserve already-passing surfaces, avoid unrelated rewrites, and justify any out-of-scope path changes in `manifest.json.risk_notes`.
