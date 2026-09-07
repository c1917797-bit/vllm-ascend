#!/usr/bin/env python3
"""Prepare and replay the frozen DRRQR calibration corpus.

Preparation tokenizes exactly sixteen source-ordered LongBench-v2 records to
2048 tokens with the target model tokenizer. Replay sends those immutable token
IDs to the OpenAI-compatible completion endpoint one request at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload)
    tmp.replace(path)


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    source_path = Path(args.dataset)
    source_bytes = source_path.read_bytes()
    records = json.loads(source_bytes)
    if not isinstance(records, list) or len(records) < args.samples:
        raise ValueError("LongBench-v2 source does not contain enough rows")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    rows = []
    for index, record in enumerate(records[: args.samples]):
        text = (
            f"{record['context']}\n\nQuestion: {record['question']}\n"
            f"A. {record['choice_A']}\nB. {record['choice_B']}\n"
            f"C. {record['choice_C']}\nD. {record['choice_D']}\nAnswer:"
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < args.seq_len:
            raise ValueError(
                f"row {index} has only {len(token_ids)} tokens; need {args.seq_len}"
            )
        token_ids = token_ids[: args.seq_len]
        token_bytes = json.dumps(token_ids, separators=(",", ":")).encode()
        rows.append(
            {
                "index": index,
                "source_id": record["_id"],
                "source_dataset_sha256": sha256_bytes(source_bytes),
                "tokenizer_config_sha256": sha256_bytes(
                    (Path(args.model_path) / "tokenizer_config.json").read_bytes()
                ),
                "token_count": len(token_ids),
                "token_ids_sha256": sha256_bytes(token_bytes),
                "token_ids": token_ids,
            }
        )
    atomic_jsonl(Path(args.output), rows)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "samples": len(rows),
                "seq_len": args.seq_len,
                "sha256": sha256_bytes(Path(args.output).read_bytes()),
            },
            indent=2,
        )
    )


def replay(args: argparse.Namespace) -> None:
    import requests

    path = Path(args.calibration)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    results = []
    for row in rows:
        if row["token_count"] != args.seq_len:
            raise ValueError(f"unexpected token count in row {row['index']}")
        response = requests.post(
            args.endpoint.rstrip("/") + "/v1/completions",
            json={
                "model": args.model_name,
                "prompt": row["token_ids"],
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
            },
            timeout=args.timeout,
        )
        body = response.content
        result = {
            "index": row["index"],
            "source_id": row["source_id"],
            "status_code": response.status_code,
            "response_sha256": sha256_bytes(body),
        }
        if response.status_code != 200:
            result["response_excerpt"] = body[:500].decode(errors="replace")
        results.append(result)
        print(json.dumps(result), flush=True)
        response.raise_for_status()
    atomic_jsonl(Path(args.output), results)
    print(
        json.dumps(
            {
                "requests": len(results),
                "successful": sum(r["status_code"] == 200 for r in results),
                "calibration_sha256": sha256_bytes(path.read_bytes()),
                "result_sha256": sha256_bytes(Path(args.output).read_bytes()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--model-path", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--samples", type=int, default=16)
    prep.add_argument("--seq-len", type=int, default=2048)
    prep.set_defaults(func=prepare)
    send = sub.add_parser("replay")
    send.add_argument("--calibration", required=True)
    send.add_argument("--endpoint", default="http://127.0.0.1:8225")
    send.add_argument("--model-name", required=True)
    send.add_argument("--output", required=True)
    send.add_argument("--seq-len", type=int, default=2048)
    send.add_argument("--timeout", type=int, default=900)
    send.set_defaults(func=replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
