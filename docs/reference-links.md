# Reference links

These are the authoritative external references for `vc4asm` and the
VideoCore IV assembler/include environment used in this repo.

## vc4asm references
- Expressions:
    - https://maazl.de/project/vc4asm/doc/expressions.html
- Directives:
    - https://maazl.de/project/vc4asm/doc/directives.html
- Instructions:
    - https://maazl.de/project/vc4asm/doc/instructions.html
- `vc4.qinc` include reference:
    - https://maazl.de/project/vc4asm/doc/vc4.qinc.html

## Working use
When changing backend lowering or emission:
1. read `docs/vc4asm-notes.md` first
2. use these links as the source of truth for assembler details
3. avoid encoding raw assembler syntax into the dialect unless the task
   explicitly requires emission-level work

## Scope reminder
The current compiler project is still in an early staged-IR phase.
These references are mainly for:
- checking terminology
- understanding helper macros
- understanding instruction categories
- planning future emission

They are **not** a reason to make the `vc4` dialect mirror raw assembler
syntax too early.