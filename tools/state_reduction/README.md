# Experimental Qwen GDN state-reduction harness

This directory contains a reproducibility harness for evaluating the Strong
RRQR/DRRQR state-reduction method from *The Key to State Reduction in Linear
Attention: A Rank-based Perspective* on Qwen3.6 packed Gated DeltaNet
checkpoints.

The code is experimental and is not wired into the vLLM Ascend runtime. The
capture installer only accepts one pinned v0.23.0 GDN source hash and is meant
for a disposable experiment container. It refuses source drift before making
an edit.

## Workflow

1. Prepare a frozen calibration corpus:

       python tools/state_reduction/state_reduction_calibration.py prepare \
         --dataset LONG_BENCH_V2_JSON \
         --model-path MODEL_DIR \
         --output calibration.jsonl

2. In a disposable container, install the bounded capture hook into the exact
   supported GDN source file:

       python tools/state_reduction/install_gdn_rrqr_capture_v023.py GDN_PY

3. Arm capture by creating `.armed` in the selected capture directory and set
   the four `VLLM_ASCEND_GDN_RRQR_*` variables embedded in the installer.
   Capture is disabled unless both the enable flag and sentinel are present.

4. Replay the immutable token IDs to the calibration server:

       python tools/state_reduction/state_reduction_calibration.py replay \
         --calibration calibration.jsonl \
         --model-name MODEL_NAME \
         --output replay-results.jsonl

5. Verify the patched hook offline:

       python tools/state_reduction/verify_rrqr_capture_hook.py --source GDN_PY

6. Build a derived checkpoint after reviewing the capture coverage:

       python tools/state_reduction/build_qwen36_drrqr_checkpoint.py \
         --src MODEL_DIR --dst OUTPUT_DIR --capture-dir CAPTURE_DIR \
         --calibration-jsonl calibration.jsonl --official-rrqr rrqr.py \
         --new-head-k-dim 64

The builder records source and calibration hashes, checks every configured
linear-attention layer and tensor, and refuses an incomplete capture set.
Use a fresh output directory on the same filesystem when possible because the
unchanged checkpoint files are hard-linked.

## Evidence boundary

The observed Dk64 and Dk96 candidates failed the frozen GSM8K quality gate.
Their performance intervals also overlapped the baseline. They must not be
presented as a validated optimization or enabled in production. See
`docs/source/developer_guide/performance_and_debug/qwen_gdn_drrqr_experiment.md`.

The full step-by-step piercing record, including raw three-run measurements,
artifact hashes, failure classification, and recovery commands, is in
`docs/source/developer_guide/performance_and_debug/qwen36_drrqr_full_piercing_report.md`.

A step-by-step methodological audit against the supplied successful HYPIC
practice is in
`docs/source/developer_guide/performance_and_debug/qwen36_drrqr_vs_hypic_audit.md`.
