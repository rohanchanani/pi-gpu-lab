Regenerate `context.txt` for this repository.

Requirements:

1. Work from the current repo root (`compiler/`).
2. Recreate `context.txt` programmatically rather than hand-editing it.
3. Preserve the current high-level format:
   - a file-tree section listing each selected file as `compiler/<path>`
   - a `Contents` section
   - one block per file using `===== compiler/<path> =====`
4. Select files from the live source tree, not from stale entries already in `context.txt`.
5. `context.txt` should contain only compiler source code and directly related build/test source files.
6. Include source files discovered programmatically, for example with `git ls-files --cached --others --exclude-standard`, `find`, `printf`, `cat`, or equivalent tools.
7. Include:
   - `CMakeLists.txt`
   - files under `include/`, `lib/`, `test/`, and `tools/`
   - source-oriented file types such as `.h`, `.cpp`, `.td`, `.mlir`, `.py`, and `.in`
8. Exclude anything that is not compiler source, including:
   - all Markdown files such as `*.md`
   - `AGENTS.md`
   - everything under `docs/`
   - binaries and other non-source assets
9. Exclude generated/build artifacts and recursive snapshot output:
   - `context.txt`
   - `context.md` from the generated snapshot
   - files under build directories such as `build/`, `build-*`, or other generated output trees
   - OS junk like `.DS_Store`
10. Overwrite `context.txt` in place.
11. After regeneration, sanity-check that every file in the file-tree section has a matching contents block.

Keep the result concise but complete for the selected source snapshot.
