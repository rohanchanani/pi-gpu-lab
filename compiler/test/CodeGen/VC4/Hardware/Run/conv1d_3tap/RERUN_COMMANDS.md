
Rerun commands

Run from repository root after parsing the generated files:

Bash
codex exec --dangerously-bypass-approvals-and-sandbox "$(cat conv1d_3tap_codex_mechanical_prompt.md)" && rm -f conv1d_3tap_codex_mechanical_prompt.md
chmod +x compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/run.sh

rm -rf compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/objs
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*.o
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*.d
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*.elf
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*.bin
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*.list
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/*shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/shader.c
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/shader.h
rm -f compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/reference/run.log

compiler/build/bin/vc4-opt \
  compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap/input.mlir \
  --vc4-verify-emit-contract \
  --vc4-verify-scheduled-hardware-rules \
  --vc4-verify-scheduled-adjacent-hazards \
  --vc4-verify-scheduled-io-spacing \
  --vc4-verify-scheduled-peripheral-accesses \
  -o /dev/null

cmake --build compiler/build --target check-vc4

compiler/test/CodeGen/VC4/Support/run_hardware_test.sh \
  compiler/test/CodeGen/VC4/Hardware/Run/conv1d_3tap \
  reference

