#!/usr/bin/env python3
"""Install the bounded DRRQR capture hook into the image's v0.23.0 GDN.

The container image carries a known-good vLLM-Ascend tree.  This installer
patches that exact file in the ephemeral container layer instead of replacing
it with a source-tree version whose vLLM imports may differ.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys


EXPECTED_ORIGINAL_SHA256 = (
    "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816"
)
MARKER = "_RRQR_CAPTURE_COUNTS: dict[tuple[int, str], int] = {}"

IMPORT_ANCHOR = "import torch\n"
IMPORT_REPLACEMENT = "import os\nimport threading\nimport time\n\nimport torch\n"

CLASS_ANCHOR = "\n\nclass AscendGatedDeltaNetAttention(GatedDeltaNetAttention):\n"
HELPER = r'''

_RRQR_CAPTURE_COUNTS: dict[tuple[int, str], int] = {}
_RRQR_CAPTURE_LOCK = threading.Lock()


@torch.compiler.disable
def _maybe_capture_rrqr_qk(
    conv_output: torch.Tensor,
    *,
    prefix: str,
    tp_rank: int,
    local_key_dim: int,
    head_k_dim: int,
) -> None:
    """Capture a bounded post-convolution Q/K sample for offline DRRQR."""
    if os.getenv("VLLM_ASCEND_GDN_RRQR_CAPTURE_ENABLE") != "1":
        return
    capture_dir = os.getenv("VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR")
    if not capture_dir or conv_output.numel() == 0:
        return
    if not os.path.isfile(os.path.join(capture_dir, ".armed")):
        return

    max_tokens = int(
        os.getenv("VLLM_ASCEND_GDN_RRQR_MAX_TOKENS", "256")
    )
    max_captures = int(
        os.getenv("VLLM_ASCEND_GDN_RRQR_MAX_CAPTURES_PER_LAYER", "1")
    )
    if max_tokens <= 0 or max_captures <= 0:
        return

    capture_key = (tp_rank, prefix)
    with _RRQR_CAPTURE_LOCK:
        capture_index = _RRQR_CAPTURE_COUNTS.get(capture_key, 0)
        if capture_index >= max_captures:
            return
        _RRQR_CAPTURE_COUNTS[capture_key] = capture_index + 1

    num_tokens = conv_output.shape[0]
    if num_tokens > max_tokens:
        sample_idx = torch.linspace(
            0,
            num_tokens - 1,
            steps=max_tokens,
            device=conv_output.device,
        ).long()
        sampled = conv_output.index_select(0, sample_idx)
    else:
        sampled = conv_output

    q = sampled[:, :local_key_dim].detach().to(device="cpu")
    k = sampled[
        :, local_key_dim : 2 * local_key_dim
    ].detach().to(device="cpu")
    os.makedirs(capture_dir, exist_ok=True)
    safe_prefix = prefix.replace("/", "_").replace(".", "_")
    path = os.path.join(
        capture_dir,
        (
            f"rank{tp_rank}_{safe_prefix}_capture{capture_index}_"
            f"{time.time_ns()}.pt"
        ),
    )
    torch.save(
        {
            "prefix": prefix,
            "tp_rank": tp_rank,
            "local_key_dim": local_key_dim,
            "head_k_dim": head_k_dim,
            "capture_index": capture_index,
            "q": q,
            "k": k,
        },
        path,
    )
'''

CALL_ANCHOR = """        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
"""
CALL_REPLACEMENT = """        else:
            mixed_qkv_non_spec = None

        if attn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None:
            capture_qkv = mixed_qkv_non_spec
            if spec_sequence_masks is None and attn_metadata.num_decodes > 0:
                capture_qkv = capture_qkv[attn_metadata.num_decode_tokens :]
            _maybe_capture_rrqr_qk(
                capture_qkv,
                prefix=self.prefix,
                tp_rank=self.tp_rank,
                local_key_dim=self.key_dim // self.tp_size,
                head_k_dim=self.head_k_dim,
            )

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
"""


def replace_once(source: str, anchor: str, replacement: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one patch anchor, found {count}: {anchor[:80]!r}"
        )
    return source.replace(anchor, replacement, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} TARGET_GDN_PY")
    target = pathlib.Path(sys.argv[1])
    raw = target.read_bytes()
    original_hash = hashlib.sha256(raw).hexdigest()
    source = raw.decode("utf-8")
    if MARKER in source:
        print(f"capture hook already installed: {target}")
        return
    if original_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unexpected GDN source: "
            f"expected={EXPECTED_ORIGINAL_SHA256} actual={original_hash}"
        )

    source = replace_once(source, IMPORT_ANCHOR, IMPORT_REPLACEMENT)
    source = replace_once(source, CLASS_ANCHOR, HELPER + CLASS_ANCHOR)
    source = replace_once(source, CALL_ANCHOR, CALL_REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source, encoding="utf-8")
    patched_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    print(
        f"installed DRRQR capture hook: original={original_hash} "
        f"patched={patched_hash} target={target}"
    )


if __name__ == "__main__":
    main()
