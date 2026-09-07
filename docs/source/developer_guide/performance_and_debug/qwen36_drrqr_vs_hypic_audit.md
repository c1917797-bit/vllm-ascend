# Qwen3.6 DRRQR piercing versus the successful HYPIC practice

## Executive answer

Not every part of the DRRQR experiment supports the same strength of claim.

- The **execution and evidence chain** is strong: identities were pinned,
  resources were isolated, the algorithm was not replaced by a proxy, all
  accepted performance and quality runs were repeated, artifacts were hashed,
  and decision gates were applied as preregistered.
- The **mechanism-feasibility claim** withstands review: the experiment
  captured real Q/K activations, produced structurally audited checkpoints,
  loaded real weights, and served a real completion.
- The **paper-exact claim does not withstand review** and is explicitly not
  made: FineWeb-Edu was unavailable.
- The **benefit claim does not withstand review**: Dk64 and Dk96 failed GSM8K,
  while their performance ranges overlapped the baseline.
- The **performance-cause claim is unavailable**: profiling was correctly
  skipped after quality failure.

The colleague's report is valuable as a workflow template, but it is not the
numeric baseline for this experiment. The two experiments use different
algorithms, models, accelerators, card counts, tensor-parallel degrees, and
treatment semantics.

## Comparison boundary

| Dimension | Colleague's successful practice | Current experiment | Comparable? |
| --- | --- | --- | :---: |
| Algorithm | HYPIC v0.8.5, position-independent prefix caching | Strong RRQR/DRRQR state reduction | no |
| Model | Qwen3.5-27B | Qwen3.6-27B | no |
| Device | 2 x Ascend 910C | 4 x Ascend 910B4 | no |
| Parallelism | TP2 | TP4 | no |
| Main treatment | HYPIC-only versus prefix-cache-disabled baseline | Dk64/Dk96 checkpoint versus Dk128 baseline | no |
| Performance shape | 32768 input, 1024 output, 40 requests, concurrency 16 | same | yes |
| Quality repetitions | three | three | yes |
| Quality suite | NIAH+, LongBench-v2, GSM8K, CEval | LongBench-v2 and GSM8K | partial |
| Performance evidence supplied | aggregate table | three raw runs plus artifact hashes | partial |
| Quality evidence supplied | aggregate table; report states three runs | three raw runs for every arm/dataset | partial |

Absolute latency, throughput, and accuracy must not be pooled across the two
experiments. Relative directions may be discussed only with the differences
above attached.

## Step-by-step execution audit

The verdict vocabulary is:

- **Strong:** the recorded evidence supports the stated claim directly.
- **Conditional:** execution is valid, but a limitation narrows the claim.
- **Failed objective:** the step executed correctly but the candidate did not
  meet its acceptance criterion.
- **Not eligible:** a later step was intentionally blocked by an earlier gate.

### Step 0: formulate the scientific question

Colleague:

- Compared HYPIC-only with a no-prefix-cache baseline.
- Used a four-group design to separate native/PDC and HYPIC switches.

Current:

- Compared the unchanged Dk128 checkpoint with Dk64.
- Preregistered Dk96 as an independent fallback only after Dk64 quality
  failure.
- Defined quality, serving, repeat-separation, and profiling gates before
  treatment measurement.

Audit:

- Current verdict: **Strong**.
- Improvement over the supplied colleague report: the acceptance formula and
  fallback identity are explicit.
- Limitation: the colleague's four-group causal design is broader. The DRRQR
  experiment isolates checkpoint identity well, but it is not a native-op x
  algorithm factorial.

### Step 1: pin the environment and treatment identity

Colleague:

- Recorded the container tag, HYPIC version, repository branch, and a package
  commit.
- The supplied result excerpt does not contain hashes for the image, model,
  dataset, or individual output artifacts.

Current:

- Pinned the paper repository commit, `rrqr.py` hash, vLLM Ascend revision,
  capture-source hash, image ID/digest, model config/index hashes, tokenizer
  hash, dataset hashes, and generated token-ID hash.
- Explicitly rejected L1 pruning and fixed-first-channel speed-oracle
  checkpoints as DRRQR evidence.

Audit:

- Current verdict: **Strong**.
- Current identity/provenance is more independently auditable than the
  supplied colleague excerpt.
- This does not imply that the colleague's result is false; it means the
  attached summary alone is insufficient to independently reconstruct every
  bit-level input.

### Step 2: isolate resources

Colleague:

- Used two logical devices for TP2.
- The example container was privileged and mounted all accelerator devices,
  which may be acceptable on a dedicated host but is unsafe to copy onto a
  shared host unchanged.

Current:

- Authorized host NPU4-7 and CPU0-95 only.
- Mapped only those four host cards into the container as devices 0-3.
- Protected NPU0-3 and the unrelated Qwen3.5 service.
- Reserved the service and HCCL port ranges and verified release at the end.

Audit:

- Current verdict: **Strong**.
- The useful colleague lesson was logical device selection; the all-device
  privileged container recipe was deliberately not reused.

### Step 3: freeze workload and evaluation semantics

Colleague:

- Performance shape: 32768/1024, 40 requests, concurrency 16.
- Quality: batch size 32, temperature 0, thinking disabled,
  `ignore_eos=false`, `trust_remote_code=false`, and non-reasoning-content
  extraction.
- Quality was run three times.

Current:

- Reused the same performance shape.
- Froze two performance warm-ups and three measured repetitions.
- Froze seed 0, temperature 0, thinking disabled, `ignore_eos=false`, and
  three quality repetitions.
- Used separate performance and quality service lifecycles.

Audit:

- Current verdict: **Strong** for the shared settings.
- Exact evaluator implementations still differ, so matching request shape
  does not make the measurements numerically interchangeable.

### Step 4: prepare algorithm-specific inputs

Colleague:

- HYPIC is a runtime prefix-cache method and does not require DRRQR activation
  calibration or checkpoint surgery.

Current:

- The paper's FineWeb-Edu calibration corpus was unavailable.
- Froze 16 source-ordered LongBench-v2 rows at exactly 2048 target-model token
  IDs.
- Recorded the source and token-ID hashes.

Audit:

- Current verdict: **Conditional**.
- The input is deterministic and suitable for testing the adapted mechanism.
- It cannot establish paper-exact reproduction or explain whether FineWeb-Edu
  would preserve GSM8K better.
- This is the largest fidelity gap in the current experiment.

### Step 5: prove algorithm activation

Colleague:

- The supplied practice lists HYPIC environment variables and an evidence-file
  switch.
- The excerpt does not include the produced evidence file or a runtime
  activation audit.

Current:

- Captured post-convolution Q/K from 48 real linear-attention layers, four TP
  ranks, and 16 samples: 3072 files.
- Required both an enable flag and an `.armed` sentinel.
- Verified per-layer, per-rank, and per-capture-index coverage.
- Selected channels with the pinned official Strong RRQR rule.

Audit:

- Current verdict: **Strong** for DRRQR mechanism activation.
- The capture hook intentionally copies bounded tensors to CPU; it was
  disabled during performance measurement.

### Step 6: build and structurally audit the treatment

Colleague:

- Installed the HYPIC plugin as an editable package.
- Treatment was configured through runtime variables.

Current:

- Dk64 and Dk96 each changed all 48 intended linear-attention layers, 96
  tensors, and six shards.
- The converter derived layer IDs from `text_config.layer_types`, validated
  capture coverage, rewrote shards atomically, and emitted a manifest.
- Dk64 manifest:
  `2eaa9a14fffe3dc0cd37906e08bee857be7c9ce2ed761f693a99a0ef02d90c0d`.
- Dk96 manifest:
  `a8f8f162f15fb6e4717ed509927d93c549a2ef01eca4c88e45e75cdf900b16fc`.

Audit:

- Current verdict: **Strong**.
- An earlier all-layer assumption failed before measurement; deriving the
  hybrid-layer map from configuration was the accepted repair.

### Step 7: load real weights and run a smoke test

Colleague:

- The reported benchmark and quality results imply a live serving path.
- The supplied excerpt does not provide a separate, hashed smoke artifact.

Current:

- Dk64 health and completion both returned HTTP 200.
- Exact response: `ASCEND_DRRQR_OK`.
- Runtime log identified the GDN prefill path with `head_k_dim=64`.
- Accepted operation: `exec-0000000000000aa5`.

Audit:

- Current verdict: **Strong** for load/runtime feasibility.
- A smoke test does not prove quality or performance; those remained separate
  gates.

### Step 8: measure baseline variance

Colleague:

- Reported baseline aggregates and stated that quality ran three times.
- The supplied document calls small changes “within noise” but does not show
  the per-run baseline ranges or a numeric noise rule.

Current:

- Stored all three raw performance runs and artifact hashes.
- Stored all three raw quality runs.
- Measured LongBench-v2 baseline range as 3.937008 pp.
- Measured GSM8K baseline range as 0.151630 pp.
- Used those ranges as dataset-specific tolerances.

Audit:

- Current verdict: **Strong**.
- Current noise handling is more testable than the qualitative “within noise”
  wording in the supplied colleague report.

### Step 9: measure performance

Reported relative changes:

| Metric | Colleague HYPIC | DRRQR Dk64 | DRRQR Dk96 |
| --- | ---: | ---: | ---: |
| TTFT | -46.6% | -2.717% | -0.401% |
| TPOT | -18.4% | -2.620% | -0.443% |
| Output throughput | +38.5% | +2.711% | +0.381% |
| Batch duration | -27.8% | -2.600% | -0.327% |

Audit:

- Current measurement execution verdict: **Strong**: three accepted runs,
  40/40 requests, zero failures, exact output length, artifact hashes.
- Current performance-benefit verdict: **Conditional/unsupported** because
  every treatment range overlapped the baseline.
- The colleague gains are much larger, but they belong to another algorithm
  and denominator. They do not predict DRRQR gains.
- The supplied colleague summary lacks raw performance repetitions, so its
  run-to-run separation cannot be independently checked from that document
  alone.

### Step 10: measure quality

| Dataset | Colleague HYPIC change | DRRQR Dk64 change | DRRQR Dk96 change |
| --- | ---: | ---: | ---: |
| NIAH+ | 0.00 pp | not run | not run |
| LongBench-v2 | +1.05 pp | +2.624672 pp | +3.674541 pp |
| GSM8K | -0.17 pp | -0.202173 pp | -0.303260 pp |
| CEval | +0.12 pp | excluded | excluded |

Audit:

- Current data-integrity verdict: **Strong** for LongBench-v2 and GSM8K:
  three raw runs for every arm and fixed denominators.
- Current coverage verdict: **Conditional** because NIAH+ was unavailable and
  CEval was excluded by prior operator decision.
- Dk64 objective: **Failed** because 0.202173 pp GSM8K loss exceeded the
  0.151630 pp baseline range.
- Dk96 objective: **Failed** because 0.303260 pp loss exceeded the same range.
- The colleague's GSM8K loss (-0.17 pp) is similar in magnitude to Dk64's
  (-0.202173 pp), but the colleague document does not expose a baseline range.
  Its “noise” label cannot be imported into the current evaluator.

### Step 11: profile and explain the performance

Colleague:

- The supplied successful-practice excerpt reports serving outcomes but no
  same-workload native profile.

Current:

- Profiling was preregistered to start only after all quality gates passed.
- Both candidates failed; profiling was not eligible and was not started.

Audit:

- Current process verdict: **Strong** because it followed the gate.
- Causal-performance verdict: **Not eligible**. The experiment cannot claim a
  kernel cause for the small mean movements.
- A future exploratory profile could diagnose the mechanism, but it would not
  reverse the frozen adoption decision.

### Step 12: make and publish the decision

Colleague:

- Reported HYPIC-only as deployable, with large performance gains and no
  material quality loss across four datasets.

Current:

- Published the negative result:
  `reject-dk64-and-dk96-under-frozen-quality-gate`.
- Kept mechanism feasibility separate from benefit.
- Did not average LongBench-v2 improvement with GSM8K loss.
- Did not use paper H100 training numbers as an Ascend serving baseline.
- Released the experiment lane and recorded that the protected lane remained
  uninspected.

Audit:

- Current verdict: **Strong** and appropriately conservative.
- The negative conclusion is not a failed piercing process. It is a completed
  experiment whose candidates failed the acceptance criteria.

## What withstands scrutiny, and what does not

| Claim | Verdict | Reason |
| --- | :---: | --- |
| The intended DRRQR code and activation data were used | yes | Official commit/hash, full capture audit, proxy checkpoints excluded |
| Dk64/Dk96 checkpoints are structurally valid | yes | Complete layer/tensor/shard audit and manifests |
| The checkpoints run on the tested Ascend stack | yes | Real-weight health and completion smoke |
| Measurements are traceable | yes | Raw repetitions, denominators, operation IDs, artifact hashes |
| Resource isolation was respected | yes | Explicit lane boundaries and recorded release |
| This is a paper-exact reproduction | no | FineWeb-Edu replaced by frozen LongBench-v2 calibration |
| DRRQR improves quality | no | LongBench-v2 rose, but GSM8K failed |
| DRRQR delivers a stable serving speedup | no | All three-run performance ranges overlap |
| Dk64 or Dk96 should be adopted | no | Both fail the frozen quality gate |
| The small speed movement has a known kernel cause | no | Profiling was ineligible after quality failure |
| Results generalize to other models or workloads | no | One model, one hardware/runtime contract, two ranks |

## Overall methodological comparison

The most useful parts of the colleague's practice were successfully reused:

1. define a real baseline rather than comparing only against a paper number;
2. keep request shape identical;
3. restart between performance and quality;
4. repeat quality three times;
5. retain a clear deployment decision.

The current experiment strengthened that template by adding:

1. bit-level identity hashes;
2. explicit shared-machine isolation;
3. algorithm-activation and layer-coverage evidence;
4. raw three-run performance artifacts;
5. measured dataset-specific baseline noise;
6. preregistered fallback and profiling gates;
7. failure classification and resumable per-sample output;
8. an explicit negative-result path.

The current experiment is weaker in two substantive ways:

1. calibration is an adaptation rather than the paper's FineWeb-Edu setup;
2. quality coverage is two datasets rather than the colleague's four.

## Final assessment

The process is sufficiently rigorous to defend the following conclusion:

> On the frozen Qwen3.6-27B, TP4, Ascend 910B4 serving setup, the official
> Strong RRQR mechanism can be adapted and executed, but neither 50% nor 25%
> Q/K state reduction meets the measured GSM8K quality gate, and neither
> provides a repeat-separated serving benefit.

It is not sufficiently complete to defend either of these stronger claims:

- the original paper has been exactly reproduced;
- DRRQR is ineffective on all Ascend models or under FineWeb-Edu calibration.

The appropriate next scientific experiment is a new, preregistered treatment,
not reinterpretation of the rejected runs: obtain the paper calibration data
if possible, validate an alignment-compatible rank near 20% reduction, freeze
the same baseline and gates again, and keep the new result independent.
