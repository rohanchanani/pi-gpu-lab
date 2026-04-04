# Codex playbook

## What Codex should optimize for
- small, reversible changes
- architecture clarity
- narrow, testable milestones
- preserving existing VC4 experimental materials

## What Codex should read first
1. `AGENTS.md`
2. `docs/project-scope.md`
3. `docs/architecture.md`
4. `docs/task-board.md`

## Good tasks for Codex
- create scaffolding
- write or refine narrowly scoped docs
- add minimal MLIR dialect setup
- implement one lowering pass
- add text-based tests
- refine repo structure without changing project scope

## Bad tasks for Codex
- “implement the compiler”
- “support CUDA”
- “design the whole backend”
- “add shared memory and synchronization”
- “optimize everything”

## Prompting style
Good prompts should:
- define one concrete milestone
- state what is out of scope
- name the files likely to change
- ask for a short summary of assumptions and next steps

## Preferred early milestones
1. repo/bootstrap docs
2. compiler scaffold
3. minimal MLIR setup
4. tiny `vc4` dialect scaffold
5. scalar-to-vec16 lowering for a SAXPY-shaped example
6. vec16-to-`vc4` toy lowering
7. readable text emission

## Expectations for code changes
For nontrivial edits, Codex should:
- keep docs aligned with code
- avoid broad speculative abstractions
- add or update tests where possible
- leave behind clear next-step suggestions