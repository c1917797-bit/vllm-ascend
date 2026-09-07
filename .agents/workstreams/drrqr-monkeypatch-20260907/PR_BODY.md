### What this PR does / why we need it?

Draft for internal review in **c1917797-bit/vllm-ascend**, not an upstream
submission. This is an incremental PR stacked on the experimental harness
branch from #1; please review that dependency separately.

Implements an opt-in load-time DRRQR monkeypatch for the dense Qwen3.5-style
hybrid GDN architecture:

- Generate immutable per-head Q/K selection plans from validated calibration
  captures and original packed BF16 checkpoint headers, reusing the audited
  Strong RRQR port and pinned official source provenance.
- Apply driver hf-overrides before hybrid cache sizing/model construction.
- Wrap Qwen3_5Model construction and weight loading in every standard worker;
  crop Q/K and matching convolution before TP slicing, with exact target
  coverage checks. Preserve V, Z/gates, full attention and other tensors.
- Preserve the no-plan baseline path. No installed gdn.py edit, offline
  checkpoint rewrite, or repeated per-token channel selection is required.
- Guard the 128-only fused prefill route for reduced shapes, retaining state
  layout and the corresponding layer's prebuilt chunk metadata. Fail closed
  on missing/conflicting metadata; include the 65+65-token/four-chunk regression.
- Document setup, three retained dimensions, evidence limits, recovery and
  outstanding cache/kernel gates.

### Does this PR introduce _any_ user-facing change?

Yes, explicit opt-in via the generated hf-overrides JSON and absolute hashed
plan path. Initial support is local original safetensors, dense qwen3_5_text,
BF16, TP dividing both head counts, PP/context parallelism=1. Quantization,
LoRA, speculative/MTP and other loaders are rejected. No new environment
variables or default-on pruning.

The requested Qwen3.8-27B download/config has **not** been verified. It must
actually match the checked architecture. Dk64/90/102 correspond to removing
50%/29.6875%/20.3125% when the original Dk is128, not an equivalent fraction
of total model cost. None of these shapes is claimed NPU-validated here.

### How was this patch tested?

31 CPU tests passed on Windows Python3.9.13, torch2.8.0+cpu,
scipy1.13.1, safetensors0.7.0:

```bash
python -I -B tests/ut/tools/test_prepare_drrqr_plan.py        # 12 passed
python -I -B tests/ut/tools/test_drrqr_monkeypatch.py         # 16 passed
python -I -B tests/ut/tools/test_state_reduction_tools.py     # 3 passed
```

Ruff check and format-check passed for all six changed Python files.
markdownlint-cli0.45.0 passed for the guide and README. `git diff --check`
passed. `bash format.sh ci` could not run because the existing Windows checkout
script has CRLF line endings; full Linux lint remains a follow-up gate.

Tests use real CPU tensors and synthetic captures, but upstream model classes
and the GDN kernel are interface doubles. Spawn coverage is installation
against these doubles, **not a real vLLM engine or Ascend worker**. The bundled
Python3.12 torch import failed with c10.dll/WinError1114; it was not counted as
a tested environment. Detailed evidence is recorded under
`.agents/workstreams/drrqr-monkeypatch-20260907/evidence/`.

### Required before production or upstream review

- [ ] Verify the actual target model and independent shard/capture provenance.
- [ ] Validate a capture path for the newer runtime; the legacy v0.23 source
  editing capture installer is not applicable to this code base.
- [ ] Verify cache page alignment and every requested kernel shape on NPU;
  Dk90/102 may be rejected before weight loading by existing alignment checks.
- [ ] Run real model startup, prefill/decode, multi-request/chunked-prefill and
  numerical correctness checks on explicitly reserved hardware.
- [ ] Run paired same-machine baseline/treatment and profiling, then report
  performance and held-out quality loss, including unsupported/slower outcomes.
- [ ] Run Linux lint/full relevant integration CI.

No NPU was used or stopped in this implementation step. NPU4-7 workloads
remain untouched. No performance gain, accuracy preservation or reproduction
of the prior HYPIC case is claimed. Keep this PR in Draft until the relevant
integration gates have evidence.

- vLLM version: source-interface review only; no installed engine was executed.
- vLLM main: e6bfe03ad73a3330cb427885aa90d97a12e1c704
