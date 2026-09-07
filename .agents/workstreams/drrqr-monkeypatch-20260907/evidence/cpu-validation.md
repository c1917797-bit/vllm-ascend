# CPU validation: 2026-09-07

Scope: this development worktree only. No remote model/server command, NPU
allocation, workload stop, benchmark or profiling was performed.

## Environment and observed results

- Windows system Python 3.9.13.
- torch 2.8.0+cpu, scipy 1.13.1, safetensors 0.7.0.
- Ruff 0.16.6 in isolated ignored .venv-drrqr.

| Command | Observed result |
| --- | --- |
| python -I -B tests/ut/tools/test_prepare_drrqr_plan.py | 12 tests, OK |
| python -I -B tests/ut/tools/test_drrqr_monkeypatch.py | 16 tests, OK |
| python -I -B tests/ut/tools/test_state_reduction_tools.py | 3 tests, OK |
| Ruff check on six changed Python files | All checks passed |
| Ruff format --check on six changed Python files | 6 files already formatted |
| markdownlint-cli 0.45.0 on README and monkeypatch guide | Exit 0 |
| bash format.sh ci | Blocked by CRLF in existing Windows checkout script |

The first bundled Python 3.12 environment could not import the installed CPU
torch (c10.dll initialization, WinError 1114). Tests were therefore run with
the existing working system Python above, not counted as passes on Python 3.12.
The Markdown command emitted a transitive Node engine-version warning but
returned success. Full Linux lint, full vLLM test suite and NPU tests remain
unverified.

## Evidence meaning

The plan tests use synthetic captures and a tiny safetensors model, with an
explicitly mocked official-source hash fixture. Runtime tests use real CPU
tensors but fake upstream model interfaces. The spawned process calls the
real install function against those doubles; it is not a real Ascend worker.
The TP check compares reference head-wise slices, not distributed collectives.

The 65+65-token regression checks that four prebuilt chunks and the exact
metadata object reach the fallback. It does not execute the AscendC kernel.
The missing-metadata bug was caught in source review before NPU execution.

No baseline/treatment/profiler dataset was generated in this step. Historical
Qwen3.6 and colleague HYPIC results must not be relabeled as results of this
patch. The implementation is ready for Draft review, not production approval.
