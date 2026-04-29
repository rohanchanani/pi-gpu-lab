
Rerun commands

Run these from the repository root after parsing the replacement files:

Bash
codex exec --dangerously-bypass-approvals-and-sandbox "$(cat stencil2d_5point_naive_codex_mechanical_prompt.md)" && rm -f stencil2d_5point_naive_codex_mechanical_prompt.md

rm -rf compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/objs
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*.o
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*.d
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*.elf
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*.bin
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*.list
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/*shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/reference/run.log

compiler/build/bin/vc4-opt \
  compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive/input.mlir \
  --vc4-verify-emit-contract \
  --vc4-verify-scheduled-hardware-rules \
  --vc4-verify-scheduled-adjacent-hazards \
  --vc4-verify-scheduled-io-spacing \
  --vc4-verify-scheduled-peripheral-accesses \
  -o /dev/null

cmake --build compiler/build --target check-vc4

compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/stencil2d_5point_naive \
  reference

