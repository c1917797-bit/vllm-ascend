# Qwen3.6 GDN state-reduction experiment

## Status

**Decision: reject Dk64 and Dk96 for the frozen workload.**

This record documents an experimental Ascend adaptation of the Strong
RRQR/DRRQR method from *The Key to State Reduction in Linear Attention: A
Rank-based Perspective* (arXiv:2602.04852v2). It is not a production feature
proposal and does not claim a validated performance gain.

The experiment used Qwen3.6-27B with tensor parallel size 4 on Ascend 910B4.
The official method implementation was pinned to commit
`919d8667d951c385e08510bc1267c2e7049a4f56`; its `rrqr.py` SHA-256 was
`fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62`.

## What was established

- The packed Q/K and convolution tensors could be pruned consistently.
- Dk64 and Dk96 checkpoints passed structural audits and real-weight smoke
  tests.
- This establishes mechanism feasibility only. It does not establish an
  acceptable quality/performance trade-off.

## Frozen quality gate

| Candidate | GSM8K accuracy | Delta vs baseline | Decision |
| --- | ---: | ---: | --- |
| Baseline | 96.310336% | -- | reference |
| Dk64 | 96.108163% | -0.202173 percentage points | reject |
| Dk96 | 96.007076% | -0.303260 percentage points | reject |

The baseline three-run range was 0.151630 percentage points. Both candidates
lost more accuracy than that observed baseline range, so profiling and
production integration were intentionally stopped.

## Performance observations

These are descriptive means, not validated gains, because candidate and
baseline intervals overlapped.

| Candidate | TTFT | TPOT | Throughput | Duration |
| --- | ---: | ---: | ---: | ---: |
| Dk64 vs baseline | -2.717% | -2.620% | +2.711% | -2.600% |
| Dk96 vs baseline | -0.401% | -0.443% | +0.381% | -0.327% |

Negative latency/duration values and positive throughput values favor the
candidate, but the overlap prevents attributing these changes to DRRQR.

## Reproducibility limits

- The paper's FineWeb-Edu calibration corpus was unavailable. The adaptation
  used 16 source-ordered LongBench-v2 records, each truncated to 2048 tokens.
  Therefore this is not a paper-exact reproduction.
- The result is specific to the frozen model, software, device, calibration,
  accuracy, and performance setup. It must not be generalized to other models
  or workloads without a new baseline and quality gate.
- No profiler run was performed after the quality failure. There is no
  kernel-level causal evidence for the small timing movements.
- Raw datasets, model weights, checkpoints, service addresses, and machine
  paths are deliberately excluded from the repository.

## Safe reuse

The scripts under `tools/state_reduction` preserve the reusable parts of the
experiment: deterministic corpus preparation, bounded and opt-in Q/K capture,
source-drift checks, coverage audits, checkpoint transformation, and evidence
hashes. Any follow-up should start with a less aggressive rank, pre-register
its acceptance threshold, and repeat baseline variance measurement before
profiling.
