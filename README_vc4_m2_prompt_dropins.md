# VC4 M2 prompt drop-ins

Install from the repo root:

```bash
unzip -o vc4_m2_prompt_dropins.zip
chmod +x pro_scripts/apply_vc4_m2_prompt_contract_handoff_fix.py
python3 pro_scripts/apply_vc4_m2_prompt_contract_handoff_fix.py "$PWD"
python3 -m py_compile pro_scripts/vc4_codegen_prompt_render.py pro_scripts/apply_vc4_m2_prompt_contract_handoff_fix.py
```

The installer rewrites `pro_scripts/vc4_codegen_prompt_render.py`, writes `pro_scripts/prompts/vc4_codegen_m2/m1_handoff.md`, and ensures the handoff file is listed in `pro_scripts/vc4_codegen_m2_context_profiles.json`.
