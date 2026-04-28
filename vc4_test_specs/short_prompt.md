You are continuing development of a VC4 / Raspberry Pi QPU MLIR backend and its hardware-grounded codegen test corpus.

You are generating “minimal friction workflow” hardware-run test bundles. The user will append a new test specification after this prompt. Produce the new bundle for that specification.

Critical output-format change for this cold-chat workflow:

Do NOT create downloadable zips.
Do NOT create sandbox links.
Do NOT say you attached files.
Do NOT wrap generated files in Markdown fences.

Instead, emit generated files directly in this exact format:

BEGIN_GPTWEB_FILE token=abc123 path=<relative/path>
END_GPTWEB_FILE token=abc123

Rules for GPT_WEB_FILE blocks:

* Use token=abc123 exactly.
* Paths must be relative.
* Do not use absolute paths.
* Do not use .. in paths.
* Include a final newline in every text file.
* You may emit multiple file blocks.
* Do not put Markdown fences around file blocks.
* Do not add explanations inside file blocks.

The user will parse the GPT_WEB_FILE blocks into real files, save them into the repo, run the generated Codex mechanical prompt, then run local verification and the hardware runner. Your response must therefore include:

1. A concise statement of test purpose, why it is the right next incremental step, what hardware path it exercises, and what it does not prove yet.
2. GPT_WEB_FILE blocks for the material files you author.
3. A GPT_WEB_FILE block for the Codex mechanical prompt.
4. A single copyable shell command block for applying/running the mechanical step and cleaning local test artifacts.
5. The standard local verifier / check-vc4 / hardware commands.
6. Do NOT generate or update catalog.json until the user later pastes a successful hardware run.

The material files you author should include, unless the user explicitly narrows the request:

* compiler/test/CodeGen/VC4/Hardware/Run//README.md
* compiler/test/CodeGen/VC4/Hardware/Run//input.mlir
* compiler/test/CodeGen/VC4/Hardware/Run//expected.json
* compiler/test/CodeGen/VC4/Hardware/Run//candidate/README.md
* compiler/test/CodeGen/VC4/Hardware/Run//reference/.gitignore
* compiler/test/CodeGen/VC4/Hardware/Run//reference/_harness.c
* compiler/test/CodeGen/VC4/Hardware/Run//reference/.qasm
* compiler/test/CodeGen/VC4/Hardware/Run//reference/_launch.c
* compiler/test/CodeGen/VC4/Hardware/Run//reference/_launch.h
* _codex_mechanical_prompt.md

Naming change for this cold-chat workflow:

* Use _harness.c instead of 3-test-.c.
* The generated Codex mechanical prompt must adapt Makefile/run.sh to build and boot the harness source named _harness.c.
* The Makefile may still produce a binary named 3-test-.bin if adapting from existing tests makes that easiest, but the source file must be _harness.c.
* The launcher files remain _launch.c and _launch.h.
* The qasm remains .qasm.
* The generated shader files should conventionally be shader.c and shader.h, matching existing reference style.

Do not assume a context tarball exists.
Do not assume paths outside pasted source files, pasted docs, or explicit user-provided paths exist.
If the user asks for dialect syntax and the dialect source/tests were not pasted, ask for those files before guessing.
If qasm syntax is uncertain, use pasted known-working qasm examples as the syntax standard; otherwise ask for a relevant working example.
Do not improvise when the user has given clear specifications or working examples.

External golden references:

* Broadcom VideoCore IV 3D Architecture Reference Guide:
  https://docs.broadcom.com/doc/12358545
* vc4asm assembler:
  https://github.com/maazl/vc4asm
  https://maazl.de/project/vc4asm/doc/index.html
* Pete Warden / Jetpac qpu-asm:
  https://github.com/jetpacapp/qpu-asm
* Pete Warden, “How to optimize Raspberry Pi code using its GPU”:
  https://petewarden.com/2014/08/07/how-to-optimize-raspberry-pi-code-using-its-gpu/
* Pete Warden / Jetpac pi-gemm:
  https://github.com/jetpacapp/pi-gemm
* Pete Warden GEMM QPU assembly source:
  https://github.com/jetpacapp/DeepBeliefSDK/blob/gh-pages/source/src/lib/pi/gemm_float.asm
* Herman Hermitage VideoCore IV resources:
  https://github.com/hermanhermitage/videocoreiv

Big picture:

* We are building a CUDA/NVIDIA-like MLIR backend for Raspberry Pi VideoCore IV QPUs.
* The current focus is hardware-grounded codegen tests and final-stage vc4 dialect code generation.
* Final codegen should eventually emit a three-file bundle per launchable kernel:
    1. qasm
    2. launcher .c
    3. launcher .h
* Candidate generated bundles are not active yet.
* The trusted reference side is the source of truth.
* Hardware execution is first-class / gold-standard for runnable tests.
* The backend should be faithful to what the vc4 dialect expresses and to the hardware behavior described by Broadcom and confirmed by hardware tests.

Locked hardware-run test contract:
Each hardware-run test lives under:

compiler/test/CodeGen/VC4/Hardware/Run//

Required shape during reference-only phase:

compiler/test/CodeGen/VC4/Hardware/Run//
README.md
input.mlir
expected.json
candidate/
README.md
reference/
.gitignore
Makefile
run.sh
mailbox.c
mailbox.h
_harness.c
.qasm
_launch.c
_launch.h
share/
vc4tmpl/template.c
vc4tmpl/template.h
vc4inc/vc4.qinc
any other copied share files required by vc4asm

Notes:

* The original locked contract used reference/3-test-.c. In this cold-chat workflow, the harness source is renamed to _harness.c, and Codex must adapt Makefile/run.sh accordingly.
* input.mlir is final-stage vc4 dialect input for future codegen.
* reference/ contains trusted hand-authored qasm/c/h and a bare-metal harness.
* expected.json is a stable semantic oracle.
* reference hardware run must print a final VC4_TEST_RESULT line.
* candidate side is disabled until codegen exists.
* catalog.json is factual only; do not add a test until the reference hardware run passes.
* Do not create broad future placeholders, XFAILs, or aspirational catalog entries.
* Do not claim candidate/codegen coverage until candidate-side generation exists.

Current committed / established test progression:

* minimal_thrend
* memory_output
* read_nop_write
* tmu_read_nop_write
* saxpy_basic
* saxpy_tmu
* saxpy_tmu_overlap
* saxpy_16
* qpu_num_register
* vpm_slice_visibility
* qpu_barrier_syncthreads
* saxpy_full
* matmul_naive
* matmul_blocked

Use existing committed tests as style/mechanical references, but do not overfit new work to them.

Important validated hardware facts and project assumptions:

* VC4 QPUs are 16-way SIMD processors.
* User QPU programs are launched through the V3D user-program scheduler path.
* Uniforms are a sequential 32-bit stream.
* Program termination uses a thread-end/program-end signal plus two following delay-slot instructions.
* Branches have three delay-slot instructions.
* Branch targets are immediate plus PC+4 when relative is set, plus an optional register-file A source value when the register-target bit is set.
* For register-based branches, vc4asm / hardware truth is decisive over ambiguous Broadcom wording.
* TMU direct memory lookups are the default for ordinary global/shared-memory-like loads.
* DMA/VPM loads should be used when intentionally testing VPM/shared-memory-style staging or reuse.
* VPM/VDW remains the standard output/writeback path in these small reference tests.
* Public launcher APIs expose semantic arguments plus runtime handle only. Do not expose qpu_id, num_qpus, raw uniform arrays, or V3D scheduler internals in launcher headers.
* qpu_id and num_qpus may be physically carried as uniform suffix words for now, but they remain conceptual execution builtins in compiler IR.
* QPU_NUMBER is available as B-regfile read address 38; ELEMENT_NUMBER is A-regfile read address 38.
* The qpu_num_register test established individual physical QPUs are runnable when using QPU reservations, but normal kernels must use logical warp IDs from uniforms, not physical QPU_NUMBER, for indexing.
* VPM slice visibility test established: on observed hardware V3D_IDENT1=0xc1102431, VPMSZ=12 KiB, QUPS=4, NSLC=3, num_qpus=12, but the generic user-QPU VPM path exposes one global 4 KiB user-visible VPM window. Slices do not get independent 4 KiB VPM windows.
* Broadcom erratum HW-2253 says BCM2835/BCM21553 user shaders cannot use all VPM; only first 64 rows are addressable, no workaround exists on this generation.
* Therefore model VC4 as one tiny CUDA-like SM with 12 QPU warp slots, 16 SIMD lanes per warp, one global 4 KiB shared-memory/VPM window, 16 semaphores, and one global mutex.
* Do not map slices to CUDA SMs. Slices are performance topology only.
* qpu_barrier_syncthreads passed same-slice, cross-slice, full 12-QPU block, 64 repeated generations, and two resident four-warp blocks with disjoint VPM rows and semaphore sets. The four-semaphore reusable barrier is locked as the first gpu.barrier / __syncthreads implementation under fully-resident block scheduling.
* Barrier-enabled blocks must be scheduled as fully resident resident-block waves; oversubscribed barrier blocks can deadlock.
* VPM/VDW setup/access stays protected by global QPU mutex until a dedicated setup-clobber test proves which setup state can be safely shared.
* V3D_ERRSTAT bit 12 (0x1000) is VCD idle and is not itself a relevant error. Existing tests use relevant error mask 0x0000efff when checking changes.

VC4-as-CUDA device model:

* multiProcessorCount = 1
* warpSize / subgroup size = 16
* max physical warps per SM = 12
* max logical threads per SM = 192
* max logical threads per block = 192
* shared memory per SM = 4096 bytes total, row-granular VPM
* semaphores per SM = 16
* recommended semaphores per barrier block = 4
* global loads = TMU direct memory lookups
* global stores = QPU registers -> VPM -> VDW DMA store
* shared memory = VPM rows
* gpu.barrier / __syncthreads = four-semaphore reusable barrier
* physical QPU_NUMBER = debug/profiling/topology only
* logical warp_id = uniform supplied by runtime
* lane_id = ELEMENT_NUMBER

Reusable four-semaphore barrier protocol:
For N = warps_per_block:
if N == 1, return.
If logical_warp_id != 0:
sem_inc(arrive)
sem_dec(release)
sem_inc(depart)
sem_dec(reset)
If logical_warp_id == 0:
repeat N-1 times: sem_dec(arrive)
repeat N-1 times: sem_inc(release)
repeat N-1 times: sem_dec(depart)
repeat N-1 times: sem_inc(reset)

VC4ASM semaphore syntax used in working qasm:

* srel  increments/releases semaphore
* sacq  decrements/acquires semaphore
  Semaphore IDs are 4-bit immediates. They cannot be read dynamically from uniforms by the semaphore instruction. If different resident blocks need different semaphores in qasm, dispatch statically by block_id or generate separate static paths. Block 0 normally uses semaphores 0..3; block 1 uses semaphores 4..7.

Memory allocation / runtime setup rule:

* Future tests must use one GPU allocation and one code copy per test/kernel bundle, not per semantic kernel call.
* Observed mailbox behavior: gpu_mem_alloc tag 0x3000c can stop returning success after exactly two alloc/lock/unlock/free cycles per boot even when prior frees succeeded.
* Therefore repeated test cases in one boot must not allocate/free on every launch.
* Use a runtime setup function such as _prepare(rt, max sizes…) that:
    * allocates once,
    * locks once,
    * copies each assembled kernel code array once into the launch state,
    * precomputes uniform stream pointers,
    * stores max sizes and scratch buffer offsets.
* Then _launch(…) should:
    * copy/update payload data into already allocated GPU-visible scratch,
    * update already allocated uniforms for current semantic arguments,
    * enqueue the already resident code address to the scheduler,
    * copy back only logical outputs.
* If a test has multiple kernels, store multiple code arrays/fields in the same single launch state and copy each once during prepare.
* Do not call mem_alloc/mem_free inside repeated semantic kernel launches.
* Prefer leaving the single allocation live until the hardware-runner reboot. If a shutdown function exists, do not loop alloc/free multiple times in a single boot.
* Runtime diagnostic fields should include runtime_allocations=1 and runtime_launches= for multi-case tests.

Launcher ABI discipline:

* The generated-style launcher C/H should not guard on kernel argument values such as n, m, k, tails, or multiples unless the public semantic API truly requires it.
* If a handwritten harness needs to constrain values, put that guard in _harness.c, not in the generated-style launcher C/H.
* If a kernel’s qasm handles only some shape, document that in README/input metadata and enforce it in the harness if needed.
* Physical uniform packing belongs inside launcher implementation.
* Public headers expose semantic args plus runtime handle only. No qpu_id, num_qpus, raw uniform arrays, hardware addresses, V3D scheduler internals, VPM rows, or semaphore IDs unless the test explicitly exists to expose a low-level runtime API.
* Uniform ABI is flat sequential 32-bit words: semantic args first, builtin suffix after, such as qpu_id and num_qpus.
* Launch policy such as active QPU count comes from runtime state, not public per-call knobs.

Harness style:

* Use bare-metal style from working tests.
* Include “rpi.h” and “_launch.h”.
* Use void notmain(void), not main.
* Use printk, not printf.
* Use panic for unrecoverable setup mistakes when appropriate.
* Generate deterministic input data.
* Run CPU reference when relevant.
* Verify results and print helpful first few mismatches.
* Use sentinel/guard regions for tail tests.
* Print stable per-case diagnostic lines.
* Print exactly one final machine-readable VC4_TEST_RESULT line at the end.
* The final result line must include name=, status=PASS or FAIL, and stable semantic fields that expected.json checks.
* Do not require elapsed_usec in expected.json.
* Do not require unstable checksums unless checksum is intentionally deterministic and stable.
* Use float tolerances in expected.json via float_max.

expected.json rules:

* Top-level name exact.
* Top-level status exact.
* required fields are exact comparisons after type coercion.
* float_max fields require abs(actual) <= limit.
* Keep it small and stable.
* Do not require timing.
* For exploratory tests, do not bake the unknown hardware result into expected.json unless the test has already established it. For example, vpm_slice_visibility did not require conclusion_id before discovery.

QASM standards:

* Use VC4ASM syntax from existing working qasm. Do not invent syntax.
* Labels use :label. Do not use C-style label: unless a working VC4ASM file proves it.
* Do not use malformed .global lines. If no external branch target/export is needed, omit .global.
* Include helper macros with .include “../share/vc4inc/vc4.qinc” only if the copied share tree is present.
* run.sh should assemble .qasm into shader.c and shader.h.
* Launcher C should include “shader.h”.
* Thread end should use the working style:
  nop; thrend
  nop
  nop
* The final three instructions (thread-end plus two delay slots) must not access uniforms, VPM, VDR, VDW, or regfile address 14.
* Thread-end instruction must not write physical regfile A/B.
* Branches have three delay slot instructions. Always fill them deliberately.
* Broadcom small immediate rule: small-immediate instructions cannot read from regfile B in the same instruction because the B read field encodes the immediate. If a loop variable is in rbN and you need immediate 1, move it through an accumulator:
  mov r3, rb21
  shl r3, r3, 1
  mov rb21, r3
  mov r3, rb20
  add r3, r3, 1
  mov rb20, r3
* Avoid immediate-read of physical regfile location written by previous instruction. Use accumulators and/or spacing.
* Do not combine semaphore instructions with closely coupled peripheral accesses.
* TMU direct memory lookup is the preferred ordinary load path. Direct memory lookup writes the absolute address to TMU0_S/TMU1_S and signals TMU read; the 32-bit result arrives in r4.
* For independent TMU loads in performance-sensitive tests, overlap them idiomatically if the test is intended to model codegen.
* For VPM horizontal 32-bit write setup row y: 0x00001a00 | (y & 0x3f).
* For VPM horizontal 32-bit read setup one vector row y: 0x00101a00 | (y & 0x3f).
* After VPM read setup, wait at least three QPU instructions before reading VPM_READ.
* Consume exactly the number of vectors requested by VPM read setup.
* Do not queue more than two VPM read blocks at a time.
* Protect VPM setup + VPM read/write + VDW setup/store/wait with the global mutex until setup-clobber testing proves a narrower rule is safe.
* VDW store path stages one horizontal VPM row, programs VDW basic setup, writes VPM_ST_ADDR, then reads VPM_ST_WAIT.
* Use dynamic VDW DEPTH for tail stores when storing only a logical partial vector.
* Use row 63 or another documented row for VDW staging only if it does not collide with shared-memory rows.

VPM/shared-memory model:

* Use V3D_VPMBASE = 16 for a 4096-byte user reservation.
* VPM row = 16 32-bit words = 64 bytes.
* User-visible VPM window = 64 rows = 4096 bytes.
* Treat VPM as one global shared memory pool across all QPUs/slices.
* Do not assume the physical 12 KiB VPM can be split as 4 KiB per slice. It cannot be configured that way on this VC4 path.
* For CUDA-like shared memory, allocate VPM rows per resident block. Multi-resident blocks must use disjoint rows and disjoint semaphore IDs.
* Hidden compiler rows may be needed for VDW staging or scratch; do not expose all 64 rows to user shared memory if codegen needs hidden rows.

Matmul and tiled-kernel guidance:

* matmul_naive shape: one QPU request is one 16-lane warp; row = qpu_id + t * num_qpus or logical warp distribution; col = column tile base + lane; use TMU global loads for A/B; accumulate in QPU f32; VDW vector stores for contiguous columns with dynamic tail depth.
* matmul_blocked shape: CUDA-like cooperative block uses VPM as shared memory. One block may compute a 12x16 output tile with 12 logical QPU warps and 16 lanes. Warps stage a B tile or other shared tile in VPM rows, use gpu.barrier/four semaphores, then consume VPM shared rows.
* For tile-wave accounting, use actual ceil formulas. Example for 12x16 output tiles: tile_waves = ceil(m/12) * ceil(n/16), excluding cases with m==0 or n==0 depending on launcher semantics. Do not hard-code wrong expected values; matmul_blocked with cases including (13,17) and (25,31) had total tile waves 14, not 13.
* Use one allocation/code copy and repeated uniform/payload updates for multi-case matmul tests.

input.mlir:

* Every input.mlir under compiler/test needs a RUN line:
  // RUN: vc4-opt %s –vc4-verify-emit-contract –vc4-verify-scheduled-hardware-rules –vc4-verify-scheduled-adjacent-hazards –vc4-verify-scheduled-io-spacing –vc4-verify-scheduled-peripheral-accesses -o /dev/null
* It must be parseable by current vc4-opt unless the user explicitly says this is an uncataloged exploratory test.
* It should describe the same semantics and launcher ABI as the reference bundle.
* Use current pasted dialect source/tests for exact op/property names. Do not guess.
* If dialect syntax is not provided and cannot be inferred from pasted working input.mlir examples, ask the user to paste the needed dialect definitions/tests.
* For exploratory hardware tests where the dialect cannot yet express a hardware operation cleanly, use a verifier-friendly metadata carrier only if current tests establish that style; document the limitation in README/input metadata.
* Do not propose broad dialect changes unless the input.mlir verifier failure is clearly due to missing dialect support, and only after inspecting pasted TableGen/C++/tests.

Codex mechanical prompt:
Generate _codex_mechanical_prompt.md as a GPT_WEB_FILE block. The prompt must be highly specified and must say Codex may only copy/adapt mechanical files:

* reference/Makefile
* reference/run.sh
* reference/mailbox.c
* reference/mailbox.h
* share/vc4inc/vc4.qinc
* share/vc4tmpl/template.c
* share/vc4tmpl/template.h
* any other files under share/ required by vc4asm templates/includes

Codex must not touch:

* compiler implementation files
* CMake files
* dialect tests
* docs unrelated to the test
* catalog.json
* material files emitted by you, except that it may not overwrite _harness.c, .qasm, _launch.c, or _launch.h
* README.md, input.mlir, expected.json, candidate/README.md, reference/.gitignore unless your generated Codex prompt explicitly says no-op verification only

Codex prompt must require:

* Copy a working donor test’s reference/Makefile, reference/run.sh, mailbox.c, mailbox.h.
* Adapt Makefile to build _harness.c.
* Adapt run.sh to assemble .qasm into shader.c and shader.h.
* Copy the entire donor share/ tree recursively into compiler/test/CodeGen/VC4/Hardware/Run//share. Do not copy only template.h; vc4asm may require template.c.
* Use the strict non-silent assembler capture pattern.

Required run.sh pattern:

#!/bin/bash
set -euo pipefail

echo “ASSEMBLING QASM”
if ! out=$(vc4asm -c shader.c -h shader.h .qasm 2>&1); then
echo “▶ ASSEMBLY FAILED WITH OUTPUT:”
printf ‘%s\n’ “$out”
exit 1
fi

if [[ -n “$out” ]]; then
echo “▶ ASSEMBLY PRODUCED UNEXPECTED OUTPUT:”
printf ‘%s\n’ “$out”
exit 1
fi

echo “RUNNING MAKE”
make

The run.sh must not power-cycle the Pi. The support runner does power cycling.

The Codex prompt should tell Codex to fail loudly if no donor share/ tree with vc4tmpl/template.c and vc4inc/vc4.qinc exists. It must not synthesize vc4asm templates.

Standard user commands to include after GPT_WEB_FILE blocks:
Use repo root path if the user has supplied one; otherwise show commands from repo root. Existing workflow often uses:
cd /Users/rohanchanani/Downloads/pi-gpu-lab

After GPT_WEB_FILE parser has written files, the command block should:

* run the generated Codex mechanical prompt:
  codex exec –dangerously-bypass-approvals-and-sandbox “$(cat _codex_mechanical_prompt.md)” && rm -f _codex_mechanical_prompt.md
* clean only this test’s transient artifacts:
  rm -rf compiler/test/CodeGen/VC4/Hardware/Run//reference/objs
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/.o
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/.d
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/.elf
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/.bin
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/*.list
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/*shader.c
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/*shader.h
  rm -f compiler/test/CodeGen/VC4/Hardware/Run//reference/run.log
* If reference/run.sh is generated by Codex, include chmod +x for it after Codex:
  chmod +x compiler/test/CodeGen/VC4/Hardware/Run//reference/run.sh

Then include standard checks:

compiler/build/bin/vc4-opt
compiler/test/CodeGen/VC4/Hardware/Run//input.mlir
–vc4-verify-emit-contract
–vc4-verify-scheduled-hardware-rules
–vc4-verify-scheduled-adjacent-hazards
–vc4-verify-scheduled-io-spacing
–vc4-verify-scheduled-peripheral-accesses
-o /dev/null

cmake –build compiler/build –target check-vc4

compiler/test/CodeGen/VC4/Support/run_hardware_test.sh
compiler/test/CodeGen/VC4/Hardware/Run/
reference

If the hardware run fails:

* Ask for the full relevant log/snippet if not provided.
* Do not redesign broadly.
* Identify whether the issue is qasm, launcher/harness, expected.json/oracle, or Codex mechanical files.
* Provide only changed GPT_WEB_FILE replacement blocks and the minimal commands to apply/re-run.
* Do not tell the user to clean broad build artifacts unless required.
* Up to three iterations of fixes are expected. Keep them surgical.

Catalog update:

* Do not emit a catalog update in the initial bundle.
* Only after user pastes a successful hardware run, provide a command to validate reference/run.log against expected.json and append/update a factual implemented_tests entry in compiler/test/CodeGen/VC4/catalog.json.
* Catalog entry fields:
  id
  name
  kind = hardware-run
  test_dir
  input_path
  reference_dir
  expected_path
  run_command
  requires_hardware = true
  reference_passed = true
  candidate_enabled = false
  notes
* Never add aspirational catalog entries.

When generating the response for the appended test specification:

* Make exactly one test bundle unless the user explicitly requests multiple.
* Keep the test focused on one understandable hardware/codegen concept.
* State purpose, incremental step, hardware path, and limitations first.
* Then emit GPT_WEB_FILE blocks.
* Then emit commands.
* Do not use downloadable links.
* Do not use Markdown fences around files.
* Ensure every text file has a final newline.
* Ensure all paths are relative and contain no ..
* Prefer correctness and evidence over cleverness.
* If the requested test needs a reference file or syntax not pasted, ask for it before guessing.