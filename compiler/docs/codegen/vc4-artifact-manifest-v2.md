# VC4 Artifact Manifest v2

Manifest v2 describes a VC4 program bundle. A one-kernel program is represented with `kernels.length == 1`; there is no separate single-kernel layout.

Required top-level keys:

```json
{
  "schema_version": 2,
  "kind": "vc4-codegen-artifact-bundle",
  "program_name": "...",
  "target": {
    "name": "vc4-bcm2835-user-qpu",
    "warp_size": 16,
    "max_active_qpus": 12,
    "shared_vpm_bytes": 4096,
    "semaphores": 16
  },
  "kernels": []
}
```

Every kernel entry must include `kernel_id`, `symbol_name`, `public_name`, `qasm_path`, `code_symbol`, `scheduled_sink_ops`, `uniform_words_per_request`, `max_requests_per_wave`, `tail_policy`, `schedule_mode`, `args`, `builtins`, and `resources`.

`qasm_path` is bundle-relative and must stay inside the bundle.

The root-level `kernel.qasm` spelling is not part of the M2 bundle contract. Every QASM artifact is addressed through `kernels[].qasm_path`; for the current emitter that path is `kernels/<public_name>.qasm`.
