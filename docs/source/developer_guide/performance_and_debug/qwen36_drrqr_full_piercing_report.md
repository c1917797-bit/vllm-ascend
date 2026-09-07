# DRRQR algorithm piercing: complete Qwen3.6-27B example

> Paper: *The Key to State Reduction in Linear Attention: A Rank-based
> Perspective* (arXiv:2602.04852v2)
>
> Model: Qwen3.6-27B
>
> Device: 4 x Ascend 910B4, tensor parallel size 4
>
> Experiment date: 2026-09-06
>
> Final state: complete; Dk64 and Dk96 rejected by the frozen quality gate

This is the complete, evidence-backed record of the experiment. It follows the
same step-by-step form as the earlier successful HYPIC piercing example while
separating three different claims:

1. **Mechanism feasibility:** established. Capture, Strong RRQR selection,
   packed-checkpoint conversion, loading, and real serving all worked.
2. **Paper-exact reproduction:** not established. FineWeb-Edu was unavailable
   and the calibration corpus was adapted.
3. **Usable Ascend benefit:** not established. Both ranks failed GSM8K and the
   performance ranges overlapped the baseline.

Internal usernames and absolute machine paths are replaced below with
environment variables. Hashes, counts, operation IDs, measurements, and
decisions are retained unchanged.

---

## Step 0. Freeze the question, resources, and decision rules

### 0.1 Scientific question

Does paper-aligned Strong RRQR reduction of the per-head Q/K state dimension
from 128 to 64 improve complete-model Qwen3.6-27B serving on four Ascend 910B4
NPUs without quality loss? If Dk64 fails quality, does the preregistered,
independent Dk96 treatment provide an acceptable 25% reduction?

### 0.2 Pinned identities

| Item | Frozen value |
| --- | --- |
| Paper implementation | `camail-official/LinearAttentionPruning` |
| Official commit | `919d8667d951c385e08510bc1267c2e7049a4f56` |
| Official `rrqr.py` SHA-256 | `fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62` |
| vLLM Ascend experiment revision | `fde48fd743ce277361e4a743a5e2e97d02912875` |
| Capture source SHA-256 | `fb476fd51b3b3456f9bee4620cabc6cf404ef78230fc9d94bb4c8661c633d82d` |
| Container image ID | `sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1` |
| Container image digest | `quay.io/ascend/vllm-ascend@sha256:471744200c5d0c768c1f6c6a6816dcae0fd551de2487d4d129b6fb750024ce6a` |
| Source config SHA-256 | `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c` |
| Source index SHA-256 | `a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b` |
| Tokenizer config SHA-256 | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |

### 0.3 Resource isolation

- Authorized experiment lane: host NPU4-7, CPU0-95, port 8225, memory limit
  48 GiB, HCCL port range 26910-27010.
- Container-visible devices: 0-3, mapped only from host NPU4-7.
- Protected lane: host NPU0-3 and the unrelated Qwen3.5 service. It was not
  inspected, stopped, restarted, or reused.
- Only experiment-prefixed containers could be mutated.
- Exit requirement: experiment containers exited, port 8225 free, and NPU4-7
  empty.

This isolation differs from the historical HYPIC command that mounted all
eight cards with `--privileged`. That historical container recipe was not
portable to a shared machine.

### 0.4 Frozen test contract

Performance:

- 32768 input tokens, 1024 output tokens;
- 40 requests, concurrency 16;
- `ignore_eos=true`;
- two warm-up runs followed by three measured repetitions;
- report TTFT, TPOT, output tokens/s, duration, E2E, and batch delay.

Quality:

- LongBench-v2: 127 rows, SHA-256
  `cdaa9e98deaefbba86f5eadeb287dd2ae90ed8f33313587dc547218b770fa39a`;
- GSM8K: 1319 rows, SHA-256
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`;
- three repetitions, seed 0, temperature 0, thinking disabled,
  `ignore_eos=false`;
- CEval was excluded by the operator before execution;
- NIAH+ was not run because no authoritative delivered dataset existed. No
  substitute was invented.

Decision rules were frozen before treatment measurement:

- Each dataset passes only when
  `baseline mean - treatment mean <= baseline three-run range`.
- A treatment cannot hide one failed dataset by averaging datasets together.
- Serving direction requires higher mean output throughput and lower mean
  duration. TTFT and TPOT must still be reported.
- A performance effect is called repeat-separated only when the complete
  three-run baseline and treatment ranges do not overlap in the beneficial
  direction.
- Profiling starts only after a candidate passes every quality gate.
- If Dk64 fails, it is rejected and Dk96 is evaluated as a separate treatment
  identity. Their results are never pooled.

---

## Step 1. Prepare the frozen calibration corpus

The paper used FineWeb-Edu with 16 batches, batch size 1, and sequence length
2048. FineWeb-Edu was not available locally. The documented Ascend adaptation
used the first 16 source-ordered LongBench-v2 records, tokenized with the target
model and truncated to exactly 2048 token IDs.

Public, equivalent command:

```bash
export REPO_ROOT=/path/to/vllm-ascend
export MODEL_DIR=/path/to/Qwen3.6-27B
export LONG_BENCH_V2=/path/to/LongBench-v2/data.json
export RUN_ROOT=/path/to/state-reduction-run
mkdir -p "${RUN_ROOT}/calibration"

python3 "${REPO_ROOT}/tools/state_reduction/state_reduction_calibration.py" prepare \
  --dataset "${LONG_BENCH_V2}" \
  --model-path "${MODEL_DIR}" \
  --output "${RUN_ROOT}/calibration/longbenchv2-16x2048-tokenids.jsonl" \
  --samples 16 \
  --seq-len 2048
```

Frozen output SHA-256:
`d146eede2a438154ea8c1bf9785dffa12f8a81e56fa370ff48be4d924e908111`.

This adaptation is sufficient to test the mechanism on the available stack,
but it is the primary reason the run must not be described as paper-exact.

---

## Step 2. Install and verify the bounded capture hook

The source-mounted vLLM Ascend implementation did not match the frozen
container ABI. The accepted solution retained the image-compatible GDN source
and installed only a minimal capture hook into the disposable container layer.
The installer:

- refuses any GDN file whose SHA-256 differs from the pinned v0.23.0 source;
- is disabled unless the enable variable equals `1`;
- also requires a `.armed` sentinel in the capture directory;
- bounds tokens and captures per layer;
- records post-convolution Q and K separately;
- writes CPU tensors only for offline selection.

```bash
export GDN_PY=/path/in/disposable/container/to/vllm_ascend/ops/gdn.py

python3 "${REPO_ROOT}/tools/state_reduction/install_gdn_rrqr_capture_v023.py" \
  "${GDN_PY}"

python3 "${REPO_ROOT}/tools/state_reduction/verify_rrqr_capture_hook.py" \
  --source "${GDN_PY}"
```

The historical launcher then replayed the frozen token IDs through a TP4
Qwen3.6 service with capture armed:

```bash
export VLLM_ASCEND_GDN_RRQR_CAPTURE_ENABLE=1
export VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR="${RUN_ROOT}/calibration/captures"
export VLLM_ASCEND_GDN_RRQR_MAX_TOKENS=2048
export VLLM_ASCEND_GDN_RRQR_MAX_CAPTURES_PER_LAYER=16
mkdir -p "${VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR}"
touch "${VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR}/.armed"

python3 "${REPO_ROOT}/tools/state_reduction/state_reduction_calibration.py" replay \
  --calibration "${RUN_ROOT}/calibration/longbenchv2-16x2048-tokenids.jsonl" \
  --endpoint http://127.0.0.1:8225 \
  --model-name qwen36-state-reduction-capture \
  --output "${RUN_ROOT}/calibration/replay-results.jsonl" \
  --seq-len 2048
```

Accepted capture audit:

```text
48 real linear-attention layer IDs
x 4 TP ranks
x 16 captures per rank/layer
= 3072 capture files
```

The exact layer IDs were derived from `text_config.layer_types`; assuming that
all model layers were linear-attention had previously caused a converter
failure.

---

## Step 3. Build the Dk64 checkpoint

Dk64 is the primary 50% Q/K state-reduction treatment. Strong RRQR selects
channels jointly from concatenated post-convolution Q/K activations for each
head. The converter prunes the packed `in_proj_qkv.weight` and
`conv1d.weight` tensors, rewrites changed shards atomically, and hard-links
unchanged files.

```bash
export IMAGE=sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1
export OFFICIAL_RRQR=/path/to/LinearAttentionPruning/src/key_reduction/pruners/rrqr.py
export DK64_DIR=/path/to/Qwen3.6-27B-gdn-drrqr-dk64

docker run --rm --network none \
  -v /path/shared/with/container:/workspace \
  "${IMAGE}" \
  python3 /workspace/vllm-ascend/tools/state_reduction/build_qwen36_drrqr_checkpoint.py \
  --src /workspace/model/Qwen3.6-27B \
  --dst /workspace/output/Qwen3.6-27B-gdn-drrqr-dk64 \
  --capture-dir /workspace/run/calibration/captures \
  --calibration-jsonl /workspace/run/calibration/longbenchv2-16x2048-tokenids.jsonl \
  --official-rrqr /workspace/LinearAttentionPruning/src/key_reduction/pruners/rrqr.py \
  --new-head-k-dim 64 \
  --expected-linear-layers 48 \
  --tp-size 4 \
  --captures-per-rank 16
```

Accepted Dk64 manifest:

- manifest SHA-256:
  `2eaa9a14fffe3dc0cd37906e08bee857be7c9ce2ed761f693a99a0ef02d90c0d`;
- 48 changed layers;
- 96 changed tensors;
- 6 changed shards;
- unchanged shards hard-linked;
- no L1 or fixed-first-channel speed-oracle checkpoint accepted as DRRQR
  evidence.

---

## Step 4. Run a real-weight Ascend smoke test

The Dk64 checkpoint was loaded by the frozen serving stack. Both health and
completion requests returned HTTP 200, and the exact canary response was
`ASCEND_DRRQR_OK`.

Runtime evidence:

```text
Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=64)
```

Operation ID: `exec-0000000000000aa5`.

This gate established structural and runtime feasibility. It did not establish
accuracy or speed.

---

## Step 5. Measure the baseline

Performance and quality used separate service lifecycles to prevent warm-up,
cache, graph, capture, and profiler state from leaking between phases:

1. start the baseline service;
2. wait for HTTP health 200 and record its configuration identity;
3. run two performance warm-ups;
4. run three measured performance repetitions;
5. stop the service;
6. restart the same baseline checkpoint and configuration;
7. wait for health again;
8. run LongBench-v2 and GSM8K three times each;
9. stop and release the service.

Historical runner invocations, with the original internal root normalized:

```bash
export WORKSTREAM_DIR=/path/to/original/experiment-workstream

python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm baseline --phase performance
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm baseline --phase quality
```

Baseline performance operation: `exec-0000000000000ab4`. All three runs
completed 40/40 requests with zero failures and exact 1024-token outputs.

| Run | TTFT ms | TPOT ms | Output tok/s | Duration s | Artifact SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 40556.828397 | 75.804468 | 120.010388 | 341.303788 | `95e54b9e144d3eacc5ccf217e677181d66a74387a738f8eb47dbe892a8bd3e98` |
| 2 | 37443.801750 | 72.544506 | 128.687989 | 318.289223 | `ee576185aeeb9158c993aebbd150a0d772ada7da584f4e4b76290370c640890f` |
| 3 | 37370.040124 | 72.537494 | 128.759025 | 318.113624 | `49c00b2f26b36f71495bcd1ab63cf1d3c12d8dc4023c30e7aa4f233e268f2394` |
| Mean | 38456.890090 | 73.628823 | 125.819134 | 325.902211 | -- |

Baseline quality operation: `exec-0000000000000acd`.

| Dataset | Run 1 | Run 2 | Run 3 | Mean | Sample SD | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 56/127 (44.094488%) | 61/127 (48.031496%) | 57/127 (44.881890%) | 45.669291% | 2.083269 pp | 3.937008 pp |
| GSM8K | 1269/1319 (96.209249%) | 1271/1319 (96.360879%) | 1271/1319 (96.360879%) | 96.310336% | 0.087544 pp | 0.151630 pp |

The measured ranges, rather than an invented universal tolerance, became the
two frozen quality thresholds.

---

## Step 6. Measure and decide Dk64

```bash
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm drrqr-dk64 --phase performance
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm drrqr-dk64 --phase quality
```

Performance operation: `exec-0000000000000b2a`.

| Run | TTFT ms | TPOT ms | Output tok/s | Duration s | Artifact SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 39843.659653 | 74.246411 | 122.245494 | 335.063475 | `d5a165c19ca9de6c8fe49521031f6b439bc211633f5b006760e9fd41113de717` |
| 2 | 36193.819163 | 70.412754 | 132.735702 | 308.583142 | `60828beb8b975f7a617ab4a864dbaada5e40117fd9914515dbb1035887a19b1d` |
| 3 | 36198.085898 | 70.439270 | 132.710317 | 308.642169 | `17bc2e3f569bb07c524a329d22cb0540771cbb75b57b5301f54213961e1ac1d8` |
| Mean | 37411.854905 | 71.699479 | 129.230504 | 317.429595 | -- |

Mean changes versus baseline:

- TTFT: -2.717%;
- TPOT: -2.620%;
- output throughput: +2.711%;
- duration: -2.600%.

The three-run intervals overlapped, so these were recorded as observed mean
movements, not a stable acceleration.

Quality operation: `exec-0000000000000b5a`.

| Dataset | Run 1 | Run 2 | Run 3 | Mean | Delta vs baseline | Baseline range | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| LongBench-v2 | 63/127 (49.606299%) | 59/127 (46.456693%) | 62/127 (48.818898%) | 48.293963% | +2.624672 pp | 3.937008 pp | pass |
| GSM8K | 1269/1319 (96.209249%) | 1267/1319 (96.057619%) | 1267/1319 (96.057619%) | 96.108163% | -0.202173 pp | 0.151630 pp | **fail** |

Decision: reject Dk64 for the frozen workload. The GSM8K loss exceeded the
observed baseline range.

---

## Step 7. Build and measure the preregistered Dk96 fallback

Dk96 reduces the per-head Q/K state by 25%. It was built only after Dk64
failed, as a new checkpoint and treatment identity.

```bash
docker run --rm --network none \
  -v /path/shared/with/container:/workspace \
  "${IMAGE}" \
  python3 /workspace/vllm-ascend/tools/state_reduction/build_qwen36_drrqr_checkpoint.py \
  --src /workspace/model/Qwen3.6-27B \
  --dst /workspace/output/Qwen3.6-27B-gdn-drrqr-dk96 \
  --capture-dir /workspace/run/calibration/captures \
  --calibration-jsonl /workspace/run/calibration/longbenchv2-16x2048-tokenids.jsonl \
  --official-rrqr /workspace/LinearAttentionPruning/src/key_reduction/pruners/rrqr.py \
  --new-head-k-dim 96 \
  --expected-linear-layers 48 \
  --tp-size 4 \
  --captures-per-rank 16
```

Dk96 manifest SHA-256:
`a8f8f162f15fb6e4717ed509927d93c549a2ef01eca4c88e45e75cdf900b16fc`.
It also changed 48 layers, 96 tensors, and 6 shards.

```bash
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm drrqr-dk96 --phase performance
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm drrqr-dk96 --phase quality
```

Performance operation: `exec-0000000000000bf2`.

| Run | TTFT ms | TPOT ms | Output tok/s | Duration s | Artifact SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 40837.282183 | 76.028506 | 119.223907 | 343.555258 | `32a7d0136b98ac66056dd897c3516e14a8cc5af7a3f87ae5c649d0650f40e694` |
| 2 | 37025.522791 | 71.945349 | 129.837447 | 315.471391 | `91bcc4632fb251f95fe8ae422ae00b20ac8a3df76cc51b27803be4ae223b3190` |
| 3 | 37045.221750 | 71.934779 | 129.833092 | 315.481972 | `9e95f1c810a44e00bd33132cddfa2b5046d5e57906708dc3c8eecf2d848fdfc6` |
| Mean | 38302.675575 | 73.302878 | 126.298149 | 324.836207 | -- |

Mean changes versus baseline:

- TTFT: -0.4010%;
- TPOT: -0.4427%;
- output throughput: +0.3807%;
- duration: -0.3271%.

All baseline and Dk96 three-run intervals overlapped.

Quality operation: `exec-0000000000000c1b`; exit code 0; duration
7,988,385 ms.

| Dataset | Run 1 | Run 2 | Run 3 | Mean | Delta vs baseline | Baseline range | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| LongBench-v2 | 60/127 (47.244094%) | 66/127 (51.968504%) | 62/127 (48.818898%) | 49.343832% | +3.674541 pp | 3.937008 pp | pass |
| GSM8K | 1265/1319 (95.905989%) | 1268/1319 (96.133434%) | 1266/1319 (95.981804%) | 96.007076% | -0.303260 pp | 0.151630 pp | **fail** |

Decision: reject Dk96 for the frozen workload.

---

## Step 8. Apply the profiling gate

Profiling was intentionally **not started** because Dk96 failed GSM8K. This is
the preregistered behavior, not missing evidence. No kernel-level cause or
operator-level speedup may be inferred from the small serving mean movements.

The paper's reported H100 training sequence-mixer result (1.32x context) is not
a same-denominator comparison with this Ascend, vLLM, complete-model serving
experiment.

---

## Step 9. Final decision and resource release

Final decision:
`reject-dk64-and-dk96-under-frozen-quality-gate`.

- Dk64 and Dk96 both failed GSM8K.
- LongBench-v2 improved in mean, but one dataset cannot cancel another
  dataset's failure.
- Dk96 showed only 0.33%-0.44% favorable mean movement, with overlapping
  intervals.
- There is no acceptable pruning ratio among the evaluated treatments for
  this frozen workload.
- Mechanism feasibility is established; paper-exact reproduction and usable
  serving benefit are not.

Final evidence audit passed. The last experiment container exited with code 0,
`OOMKilled=false`; port 8225 was free; NPU4-7 were empty; NPU0-3 remained
uninspected.

---

## Step 10. Failures, classification, and accepted recovery

Orchestration failures before measurement are not algorithm failures. Only the
repaired attempt under the unchanged frozen identity was accepted.

| Stage | Failure class | Measurement started? | Accepted recovery |
| --- | --- | --- | --- |
| Capture launcher construction | orchestration authoring | no | Apply a structured patch, run shell syntax/static checks, then relaunch under the same experiment identity. |
| Capture Docker launch | shell quoting | no | Use a Bash argument array instead of continuation-sensitive assembly. |
| Plugin editable install | offline packaging | no | Use the image-compatible minimal install path without dependency resolution. |
| Capture source mount | source/image ABI mismatch | no | Keep the image-compatible GDN and inject only the minimal bounded capture hook. |
| Checkpoint conversion | model-structure assumption | no | Derive linear layer IDs from `text_config.layer_types` and verify complete coverage. |
| Dk64 smoke launcher | shell quoting | no | Replace the command with a Bash array; accept only the later exit-0 smoke. |
| Repository validation | remote working-directory semantics | not applicable | Pass explicit Git directory and work-tree arguments, then rerun `diff --check`. |
| Read-only progress probe | remote command shape | observation only | Invoke the shell with `-lc` and pass the line as one argument. |
| Incomplete summary render | post-processing invocation | not applicable | Use the exact output root and explicitly allow incomplete rendering while treatment data are absent. |
| Performance aggregate validation | post-processing schema field | collection already complete | Inspect actual keys and rerun; verify every run is 40/40 with exact 1024-token outputs. |
| Handoff timestamp | provenance/rounding | controller still running | Read UTC from the target host and record asynchronous counts as lower bounds. |
| Dk96 converter CLI discovery | host dependency mismatch | no | Run `--help` inside the frozen image mounted on the shared data root. |
| Dk64 aggregate audit | shell interpolation | measurements complete | Pass the multiline program as a direct argument and rerun the audit separately. |
| Dk96 quality launcher | remote path error | no | Verify the runner path, controller absence, free resource lane, then relaunch the same phase. |
| Report rendering during Dk96 | incomplete-state handling | controller still running | Render raw tables only after both baseline and candidate aggregates exist. |

Reusable lessons:

- pin code, image, model, tokenizer, dataset, and generated-token identities;
- isolate devices, CPU, ports, and container names before starting;
- use real activation-based DRRQR, not an L1 proxy or speed oracle;
- keep performance and quality in separate service lifecycles;
- measure baseline variance before defining the quality tolerance;
- fail each dataset independently;
- treat a less aggressive rank as a new treatment;
- profile only a quality-qualified candidate;
- publish negative results and orchestration failures with the same rigor as
  positive results.

---

## Recovery and rerun commands

This experiment is complete, so there is no unfinished controller to resume.
The following are recovery commands for the original workstream runner, not an
instruction to bypass resource ownership:

```bash
export WORKSTREAM_DIR=/path/to/original/experiment-workstream
export RUN_ROOT=/path/to/state-reduction-run

# Read-only state inspection.
cat "${RUN_ROOT}/controller-status.json"

# Retry only missing/invalid sample rows for one already-authorized phase.
python3 "${WORKSTREAM_DIR}/state_reduction_formal_runner.py" \
  --arm drrqr-dk96 --phase quality

# These profile commands are deliberately ineligible for the recorded run,
# because Dk96 failed quality.
python3 "${WORKSTREAM_DIR}/state_reduction_native_profile.py" --arm baseline
python3 "${WORKSTREAM_DIR}/state_reduction_native_profile.py" --arm drrqr-dk96
```

Before any rerun, confirm all of the following:

1. the authoritative previous operation is terminal;
2. no matching controller or experiment container exists;
3. the explicitly authorized NPU lane is empty;
4. the configured port and HCCL range are free;
5. protected workloads are not inspected or mutated;
6. model, image, dataset, tokenizer, contract, and checkpoint hashes match;
7. profile commands remain disabled after any quality failure.

Per-sample quality JSONL is resumable: only indexes without an accepted
`ok=true` row are retried. Existing accepted rows are not overwritten.
