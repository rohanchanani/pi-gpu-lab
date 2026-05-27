You are performing a read-only implementation-integrity audit for VC4 VC4Tile M5.

Inspect the repository freely, but do not write files, stage files, or commit.

Classify whether the implementation appears legitimate or whether it cheats, special-cases, bypasses required compiler layers, hard-codes fixtures, substitutes host/reference artifacts, fakes hardware results, weakens regression contracts, lets surface ops leak past core verification, implements producer lowering in M5, or implements executable sub-32 precision.

Return exactly one JSON object and no markdown:

{"integrity_pass":"YES|NO","summary":"short explanation","findings":[{"severity":"blocking|warning","path":"repo-relative path","reason":"specific reason"}]}
