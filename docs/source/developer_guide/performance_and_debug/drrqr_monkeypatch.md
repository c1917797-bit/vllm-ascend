# Experimental DRRQR load-time monkeypatch

This is an opt-in implementation candidate, not a demonstrated speedup. It
replaces Python model construction/weight-loading functions inside each Ascend
worker. It does not edit the installed `gdn.py`, rewrite checkpoint shards, or
apply channel selection on every forward pass.

## Scope and source contract

- Paper: *The Key to State Reduction in Linear Attention: A Rank-based Perspective*.
- Algorithm source: [LinearAttentionPruning](https://github.com/camail-official/LinearAttentionPruning),
  commit `919d8667d951c385e08510bc1267c2e7049a4f56`.
- Pinned `rrqr.py` SHA256:
  `fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62`.
- Ascend base: `a9cf49deab7468e1e14c0fb5cbecf7c6155700b4`.
- Reviewed vLLM source: `e6bfe03ad73a3330cb427885aa90d97a12e1c704`, as recorded
  in `.github/vllm-main-verified.commit`. Re-review the seams after an upgrade.
- Initial contract: dense `qwen3_5_text` hybrid attention, unquantized BF16,
  original local packed safetensors, standard worker, TP dividing both head
  counts, PP=1 and context parallelism=1. LoRA, speculative/MTP execution and
  alternative checkpoint loaders are rejected.

The requested deployment model is Qwen3.8-27B. Its download and actual config
were not inspected in this implementation step. A marketing name or directory
name is not architecture evidence: the tool rejects other model types. Earlier
Qwen3.6 measurements do not validate this implementation or the new model.

## Implementation map

| Component | Responsibility |
| --- | --- |
| `tools/state_reduction/prepare_drrqr_plan.py` | Validate source headers and captures; emit per-layer, per-head selection plus hashes and driver overrides |
| `vllm_ascend/drrqr.py` | Validate immutable plan/source/config; crop packed Q/K projection and matching convolution before TP slicing |
| `vllm_ascend/patch/worker/patch_drrqr.py` | Idempotently wrap `Qwen3_5Model.__init__` and `load_weights`; guard the 128-only fused prefill route |
| `vllm_ascend/patch/worker/__init__.py` | Install wrappers in each standard Ascend worker, including newly spawned workers |

The preparation tool reuses the existing `load_keep_maps` Strong RRQR selection
port, audited against the pinned algorithm source. It selects coordinates for
each key head from calibration activations. Q and K use the same selection;
the corresponding convolution channels follow it. The V tail, Z/gates,
full-attention weights, and all other tensors pass through unchanged. This is
an adaptation to packed Qwen GDN, not a claim of reproducing the paper's models
or accuracy. Reduced Q/K are normalized using the existing GDN norm and scale.

Driver `--hf-overrides` must set the reduced dimension before cache planning
and projection construction. Patching only the worker constructor is too late
for hybrid-cache sizing. The wrapper then checks that the plan, source hashes
and effective reduced config agree before allowing model initialization.

The fused CANN availability probe exercises Dk=Dv=128. For reduced shapes the
wrapper routes prefill to the existing Triton/AscendC pipeline, preserving the
state layout and the current layer's prebuilt chunk metadata. It matches the
original cumulative-length tensor by Python identity, without device reads.
Missing or conflicting metadata raises an error: two 65-token requests need
four 64-token chunks, not `ceil(130/64)=3`. Decode kernels are not replaced.

## Reproducible preparation

Commands below are templates for the verified Linux experiment environment,
not commands executed during the CPU implementation. Supply real absolute
paths; do not substitute an older model's capture directory.

1. Verify the model download, original shard manifest, dtype, actual text
   architecture, tokenizer and dataset hashes. Record the vLLM/Ascend commits,
   CANN, driver, container and device mapping. Install this branch in an
   isolated development environment using that environment's normal Ascend
   build procedure. Do not replace a shared running installation.
2. Obtain bounded calibration captures for this exact model and record their
   origin, layer/head coverage, capture TP and dataset split. The old
   `install_gdn_rrqr_capture_v023.py` only supports its pinned v0.23 source;
   it is **not** an installer for this newer runtime. A compatible capture path
   must be validated separately before a fresh end-to-end experiment.
3. Generate a separate plan for each retained dimension. Output is exclusive:
   an existing plan is never overwritten. No derived checkpoint is created.

```bash
set -euo pipefail
set -o noclobber
: "${MODEL_DIR:?original local checkpoint}"
: "${CAPTURE_DIR:?verified captures for this model}"
: "${CALIBRATION_JSONL:?frozen calibration corpus}"
: "${OFFICIAL_RRQR:?pinned official rrqr.py}"
: "${RUN_DIR:?new experiment output directory outside MODEL_DIR}"
: "${CAPTURE_TP:?actual TP used during capture}"
: "${CAPTURES_PER_RANK:?verified bounded capture count}"
mkdir -p "$RUN_DIR"

# For original Dk=128 only. Validate a different model dimension explicitly.
for dk in 64 90 102; do
    python tools/state_reduction/prepare_drrqr_plan.py \
        --src "$MODEL_DIR" \
        --capture-dir "$CAPTURE_DIR" \
        --calibration-jsonl "$CALIBRATION_JSONL" \
        --official-rrqr "$OFFICIAL_RRQR" \
        --tp-size "$CAPTURE_TP" \
        --captures-per-rank "$CAPTURES_PER_RANK" \
        --new-head-k-dim "$dk" \
        --output "$RUN_DIR/drrqr-dk${dk}.json" \
        > "$RUN_DIR/drrqr-dk${dk}-receipt.json"
done
```

From Dk=128, these remove 50%, 29.6875%, and 20.3125% of key coordinates.
They are not claims of whole-model FLOP, memory or latency reduction. The
nearest-integer Dk90 is intentional; flooring `128*(1-0.3)` gives Dk89 and is
a different experiment. No target dimension is silently rounded by the tool.

Plan hashes bind the selection bytes; config/index hashes bind metadata, not
every shard's contents. Keep an independently audited immutable shard manifest.
Hashing a supplied calibration file cannot prove that the captures came from
it, or establish calibration/evaluation separation. Plan files are not signed;
the caller must trust the producer. They are not safe substitutes for raw
experiment evidence.

## Opt-in launch and rollback

Before any launch, reserve permitted physical devices and verify their host to
container mapping. The implementation step did not use any NPU. Existing
NPU4-7 workloads must remain untouched; never reuse an old launch script whose
logical 0-3 map to physical 4-7. Do not stop another user's process.

Only after architecture, resources and cache/kernel gates are passed, append
the emitted `hf_overrides` object to the otherwise unchanged baseline command:

```bash
: "${CONFIRMED_DEVICES:?verified reserved physical devices and container mapping}"
: "${TP_SIZE:?number of reserved devices}"
: "${PORT:?reserved experiment port}"
: "${DK:?explicit retained key dimension}"
OVERRIDES=$(python -c \
    'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["hf_overrides"]))' \
    "$RUN_DIR/drrqr-dk${DK}-receipt.json")

# Minimal interface example; keep all benchmark flags identical to baseline.
ASCEND_RT_VISIBLE_DEVICES="$CONFIRMED_DEVICES" \
vllm serve "$MODEL_DIR" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 --load-format safetensors --port "$PORT" \
    --hf-overrides "$OVERRIDES"
```

The absolute plan path must be visible at the same path in every worker's
filesystem. Do not feed a pre-pruned checkpoint into the monkeypatch. Confirm
both log messages: `DRRQR configured before weight/cache loading` and
`DRRQR weight loading complete`, with matching plan digest and tensor coverage.

Rollback: stop only this experiment's service through its recorded supervisor,
then start a **fresh process** with the original checkpoint and baseline
arguments, removing both `ascend_drrqr` and the reduced `linear_key_head_dim`
override. No source/shard restoration is needed. Live unpatching is not
supported; one service process owns one model.

## Validation and remaining gates

The standalone CPU tests intentionally avoid importing the full NPU plugin:

```bash
python -I -B tests/ut/tools/test_prepare_drrqr_plan.py
python -I -B tests/ut/tools/test_drrqr_monkeypatch.py
python -I -B tests/ut/tools/test_state_reduction_tools.py
```

They cover synthetic selection, header/hash validation, exact Dk64/90/102
cropping, reference TP1/2/4 slicing, unmodified tensor identity/content,
baseline delegation, duplicate/missing targets, idempotence, worker-spawn
installation against interface doubles, and prefill dispatch/state/metadata.
They do not execute a real vLLM engine, AscendC kernel or NPU.

Do not merge as validated acceleration until all of the following pass:

- Real target config/checkpoint and capture provenance audit.
- Cache page-size checks: the current `patch_mamba_config.py` requires exact
  SSM/full-attention page alignment. Dk90/102 may be rejected before loading;
  successful plan generation does not imply a supported cache shape. Do not
  disable that assertion or report a padded dimension as the requested ratio.
- Prefill/decode, multi-request and chunked-prefill correctness on reserved
  NPUs; kernel support for each actual shape and normal startup/warmup.
- Same-machine paired A/B with frozen requests, warmup, repeated measurements,
  throughput/TTFT/TPOT/latency and profiler attribution. Baseline must use the
  same runtime build. A shape change may route to a slower prefill kernel.
- Held-out quality measurements using the agreed datasets (CEval excluded),
  with measured loss reported even in the performance-first phase.

If a candidate is unsupported or slower, preserve the exact failure/log and
explain the limitation. No speedup, quality preservation, or completion of the
previous colleague's HYPIC experiment is claimed by this patch.
