# DRRQR monkeypatch implementation

- Scope: opt-in load-time monkeypatch for dense Qwen3.5-style hybrid GDN.
- User target: Qwen3.8-27B; download/architecture not yet verified.
- Branch: work/drrqr-monkeypatch-20260907.
- Base: a9cf49deab7468e1e14c0fb5cbecf7c6155700b4.
- vLLM source contract: e6bfe03ad73a3330cb427885aa90d97a12e1c704.
- CPU-only development. No NPU or remote runtime actions are authorized by this implementation step.
- Apply reduced config in the driver before cache planning; load-time transform precedes TP slicing.
- Current state: implementation and 31 CPU unit/regression tests complete;
  awaiting NPU validation, no hardware/performance claim.
- User selected Draft PR into c1917797-bit/vllm-ascend for internal review.
- Do not commit weights, datasets, credentials, captures, plans or the test venv.

## Delivery

- Draft PR target: c1917797-bit/vllm-ascend.
- PR base: experiment/qwen36-drrqr-state-reduction (dependency: Draft PR #1).
- PR head: work/drrqr-monkeypatch-20260907.
- Runtime: vllm_ascend/drrqr.py and patch/worker/patch_drrqr.py.
- Preparation: tools/state_reduction/prepare_drrqr_plan.py.
- User guide and failure/recovery commands:
  docs/source/developer_guide/performance_and_debug/drrqr_monkeypatch.md.
- Evidence: evidence/cpu-validation.md in this directory.

## Decisions and lessons

1. Apply dimensions through driver hf-overrides before cache planning; worker
   construction alone is too late. Both the plan and effective config are checked.
2. Transform complete Q/K and convolution weights before the original TP loader;
   never re-prune a derived checkpoint. Preserve V, Z, gates and full attention.
3. A 128-only fused probe is not evidence for reduced shapes. Preserve prebuilt
   per-request chunk metadata in the fallback; missing metadata fails closed.
4. Synthetic CPU correctness is not NPU compatibility or speedup. Cache page
   alignment and 90/102-dimensional kernel support remain unverified.
5. Source/config/index and plan hashes do not prove capture provenance or every
   shard's contents. Preserve an independent immutable shard/capture manifest.
6. Keep the legacy v0.23 capture installer separate from this pinned newer
   runtime. A new capture hook may be needed, but is outside this code-only step.

## Safe resume

```bash
git fetch origin work/drrqr-monkeypatch-20260907
git worktree add --detach ../drrqr-review origin/work/drrqr-monkeypatch-20260907
cd ../drrqr-review
python -I -B tests/ut/tools/test_prepare_drrqr_plan.py
python -I -B tests/ut/tools/test_drrqr_monkeypatch.py
python -I -B tests/ut/tools/test_state_reduction_tools.py
```

Use a fresh nonexisting review worktree path and a CPU environment with the
versions recorded in evidence. These commands do not reserve or use NPUs.
Next: inspect the real Qwen3.8 config/download, validate capture provenance and
cache shapes, then seek/respect the current resource reservation before any
canary. Physical NPU4-7 workloads remain untouched by this implementation.
Do not reuse old logical-device launch scripts without resolving their mapping.

Rollback from a future treatment run means a fresh process using the original
checkpoint and baseline arguments, removing both DRRQR metadata and reduced
dimension overrides. Stop only the recorded experiment-owned supervisor.
