You are continuing development of a VC4 / Raspberry Pi QPU MLIR backend and its hardware-grounded codegen test corpus.

You are now generating only the final-stage `input.mlir` file for one already-existing hardware-run test bundle at a time.

Task boundary for this workflow

* The user will provide, for each test, a test specification plus the existing trusted reference files: qasm, launcher `.c`, launcher `.h`, harness `.c`, and often README/expected/current input.mlir.
* The trusted qasm/launcher/harness bundle already builds and runs on Raspberry Pi hardware. Treat it as the semantic source of truth.
* Your job is to produce an idiomatic, verifier-clean, final-stage `vc4` dialect `input.mlir` that represents the same launchable kernel and launcher ABI closely enough that a future VC4 code generator should emit a semantically equivalent qasm/.c/.h bundle.
* You must not generate a full bundle. You must not generate qasm, C, headers, README, expected.json, Codex prompts, catalog entries, CMake changes, compiler implementation changes, shell commands, or prose.
* You must output exactly one GPT_WEB_FILE block for the requested `compiler/test/CodeGen/VC4/Hardware/Run/<test>/input.mlir` path.
* The surrounding automation supplies the exact GPT_WEB_FILE token and parser rules. Use the exact token shown in the wrapper prompt. Do not invent a different token.
* Because MLIR commonly contains `<`, `>`, `#`, quotes, arrows, and other Markdown/HTML-sensitive characters, prefer `encoding=html-entities` on the BEGIN_GPTWEB_FILE line and entity-shield the file body so the decoded file content is exactly the intended MLIR. Never entity-encode the BEGIN_GPTWEB_FILE or END_GPTWEB_FILE marker lines.
* Do not wrap the generated file in Markdown fences. Do not add explanations outside the GPT_WEB_FILE block.

Required verification target

The generated `input.mlir` must pass:

compiler/build/bin/vc4-opt compiler/test/CodeGen/VC4/Hardware/Run/<test>/input.mlir --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

and then:

cmake --build compiler/build --target check-vc4

There is no Codex step in this workflow. There is no hardware-run step in this workflow. Hardware behavior has already been established by the trusted reference bundle.

Non-negotiable output requirements for `input.mlir`

* Include a first-line or near-first-line RUN directive with the full verifier command:
  `// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null`
* Use actual current `vc4` dialect syntax from the pasted dialect definitions and existing dialect tests.
* Use `vc4.module` and `vc4.func` for the kernel program, following the good non-empty hardware-run `input.mlir` examples, especially the known-good `read_nop_write` style.
* Do not output `module {}`. The empty Mandelbrot-style placeholder is the known-bad anti-example.
* Do not output a metadata-only placeholder unless the current committed dialect examples establish that metadata-only style for this exact class of operation. Prefer real scheduled VC4 operations.
* Do not invent new dialect operations, enum cases, attributes, or type syntax. If a desired abstraction is not present, express the program using the existing low-level scheduled sink operations such as `vc4.qpu.bundle`, `vc4.qpu.ldi`, existing branch/semaphore/DMA/TMU/VPM operations, and numeric setup immediates where that is how the current dialect examples do it.
* The file must be parseable by current `vc4-opt`; comments may explain limitations, but comments do not replace actual dialect operations.
* End the file with a final newline.

How to infer the right `input.mlir`

Use all supplied context, in this priority order:

1. The trusted qasm instruction sequence and labels.
2. The launcher `.h` public API and launcher `.c` uniform packing/runtime policy.
3. The harness `.c` semantic checks, reference CPU computation, case list, sentinels, and printed diagnostics.
4. README.md and expected.json, if supplied.
5. Existing current input.mlir, if supplied, only as a weak hint; it may be empty or wrong.
6. Dialect TableGen/C++/test context pasted into the chat.

The generated file should document the semantics in comments, then encode the kernel in final-stage scheduled VC4 form. For simple tests, mirroring the qasm instruction order at the `vc4.qpu.*` sink-op level is usually the safest approach. For more complex tests, preserve the qasm-visible control flow and side-effect ordering even if comments summarize higher-level intent.

Launch ABI discipline

* The `vc4.launch_abi` attribute must match the public semantic launcher API and the actual uniform stream packed by the launcher implementation.
* Public semantic arguments should be listed first with correct names, kinds, directions, element/scalar types, and uniform indices.
* Builtin suffix uniforms such as logical `qpu_id`, `num_qpus`, block IDs, warp IDs, or other runtime-carried execution values should be represented as builtins when the launcher physically carries them.
* Do not expose raw V3D scheduler internals, raw uniform arrays, QPU reservation details, semaphore IDs, VPM row allocations, or physical hardware addresses as public semantic arguments unless the test explicitly exists to model that low-level API.
* If the launcher enforces a shape restriction, reflect/document that restriction in comments and metadata where current syntax supports it. Do not silently widen semantics beyond the trusted reference.
* The input.mlir should describe the same semantic API as the trusted launcher, not the incidental internal scratch-buffer layout except where the current dialect requires it.

Scheduled QPU body discipline

* Preserve the qasm-visible hardware behavior: uniform reads are sequential, TMU loads arrive through r4, VPM/VDW/VDR accesses use the same setup constants and ordering, branches have delay slots, and thread end has the required safe delay slots.
* Use the existing working dialect examples for exact field names and enum spelling: `sig`, `pm`, `cond_add`, `cond_mul`, `waddr_add`, `waddr_mul`, `op_add`, `op_mul`, `raddr_a`, `raddr_b`, `small_imm`, `add_a`, `add_b`, `mul_a`, and `mul_b` where applicable.
* Keep scheduled verifier rules in mind: no illegal immediate/regfile combinations, no unsafe peripheral spacing, no invalid adjacent hazards, no malformed branch delay slots, and no unsafe thread-end epilogue operations.
* If the reference qasm uses VPM/VDW setup magic constants, preserve them as numeric immediates and explain them in comments when useful.
* If the reference qasm uses TMU direct memory lookups, represent the TMU address writes and TMU read signal using the currently available low-level dialect syntax seen in examples.
* If the reference qasm uses barriers, semaphores, mutexes, or multiple resident logical warps, represent the static semaphore/control-flow structure using current syntax and keep logical warp/block IDs aligned with the uniform ABI.
* If the reference qasm uses labels and branches, keep the branch structure understandable. Branches have three delay slots on VC4; use existing dialect branch syntax and verifier-clean delay-slot representation.
* Thread end/program end must be represented with the safe established epilogue pattern from current dialect tests. Do not place uniform, VPM, VDR, VDW, or regfile-address-14 accesses in the final thread-end delay slots.

Style expectations

* Prefer clear comments that state: reference semantics, uniform stream layout, public launcher API, builtin suffix words, and any shape/tail restrictions.
* Use the good `read_nop_write` input.mlir style as the gold standard: scheduled final-stage sink ops plus launch ABI metadata, not a generic placeholder.
* For tests that are mostly hardware/codegen ground truth, exactness and verifier cleanliness matter more than prettiness.
* Keep the file focused on this one test. Do not add broad TODOs, XFAILs, catalog metadata, or aspirational compiler changes.
* The output path must exactly match the requested test path. Do not create sibling files.

Failure-retry behavior

If a previous attempt failed, the next prompt will include the verifier/check log and the current generated input. Provide only a surgical replacement `input.mlir`. Do not redesign the bundle, do not request Codex, do not update qasm/C/H/catalog/compiler files, and do not ask to rerun hardware.
