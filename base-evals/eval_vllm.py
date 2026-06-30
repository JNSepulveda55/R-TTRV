#!/usr/bin/env python3
"""Evaluate prepared ABCD records with a base VLM through vLLM."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams


LETTERS = "ABCD"
SUMMARY_FIELDS = [
    "timestamp",
    "dataset",
    "count",
    "correct",
    "accuracy",
    "model",
    "records",
    "output_file",
    "limit",
]


def read_jsonl(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def extract_letter(text: str) -> str:
    text = text.strip()
    explicit = re.search(
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()
    boxed = re.search(r"\\boxed\{?\s*([A-D])\s*\}?", text, flags=re.IGNORECASE)
    if boxed:
        return boxed.group(1).upper()
    standalone = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    return standalone.group(1).upper() if standalone else ""


def open_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_vllm_tokenizer(llm: LLM):
    if hasattr(llm, "get_tokenizer"):
        return llm.get_tokenizer()
    engine = getattr(llm, "llm_engine", None)
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise RuntimeError("Could not access vLLM tokenizer for prompt decoding.")


def decode_prompt_token_ids(tokenizer, prompt_token_ids) -> str:
    if prompt_token_ids is None:
        return ""
    try:
        return tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(prompt_token_ids)


def write_prompt_log(prompt_log, item: dict) -> None:
    if prompt_log is not None:
        prompt_log.write(json.dumps(item, ensure_ascii=False) + "\n")
        prompt_log.flush()


def print_prompt_log(item: dict) -> None:
    print(f"[base-eval prompt_debug] index={item['index']} id={item['id']} dataset={item['dataset']}")
    print(f"[base-eval raw_prompt] {item['raw_prompt']}")
    print(f"[base-eval vllm_output_prompt] {item['vllm_output_prompt']}")
    print(f"[base-eval decoded_prompt_token_ids] {item['decoded_prompt_token_ids']}")
    print(f"[base-eval response] {item['response']}")


def log_runtime_environment() -> None:
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1', '')}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            print(f"cuda_device={torch.cuda.get_device_name(index)}")
            print(f"cuda_capability={torch.cuda.get_device_capability(index)}")
    except Exception as exc:
        print(f"torch_runtime_probe_failed={exc}")


def update_summary_csv(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if summary_csv.exists() else "w+"
    with summary_csv.open(mode, encoding="utf-8", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        existing = list(csv.DictReader(f)) if summary_csv.stat().st_size else []
        existing = [
            old
            for old in existing
            if not (
                old.get("dataset") == row["dataset"]
                and old.get("model") == row["model"]
                and old.get("limit") == row["limit"]
            )
        ]
        existing.append(row)
        f.seek(0)
        f.truncate()
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("base-evals/results"))
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--prompt-log-file",
        type=Path,
        default=None,
        help="Optional JSONL file for debugging the prompt sent to/returned by vLLM.",
    )
    parser.add_argument(
        "--prompt-log-limit",
        type=int,
        default=0,
        help="Number of evaluated examples to write to --prompt-log-file or stdout.",
    )
    parser.add_argument(
        "--log-prompts-to-console",
        action="store_true",
        help="Print prompt debugging blocks to the Slurm stdout log.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.records, args.limit)
    if not records:
        raise RuntimeError(f"No records found in {args.records}")
    if args.prompt_log_file is not None and args.prompt_log_limit <= 0:
        raise ValueError("--prompt-log-limit must be positive when --prompt-log-file is set.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.out_dir / f"{args.dataset}.jsonl"
    summary_file = args.out_dir / f"summary_{args.dataset}.json"
    if args.prompt_log_file is not None:
        args.prompt_log_file.parent.mkdir(parents=True, exist_ok=True)

    log_runtime_environment()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = get_vllm_tokenizer(llm) if args.prompt_log_limit > 0 else None
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    correct = 0
    count = 0
    prompt_log = args.prompt_log_file.open("w", encoding="utf-8") if args.prompt_log_file is not None else None
    with output_file.open("w", encoding="utf-8") as out:
        for batch in tqdm(list(batched(records, args.batch_size)), desc=args.dataset):
            requests = [
                {
                    "prompt": record["prompt"],
                    "multi_modal_data": {"image": open_image(record["image_path"])},
                }
                for record in batch
            ]
            outputs = llm.generate(requests, sampling)
            for record, output in zip(batch, outputs):
                response = output.outputs[0].text
                pred = extract_letter(response)
                is_correct = pred == record["answer"]
                correct += int(is_correct)
                count += 1
                if count <= args.prompt_log_limit:
                    prompt_token_ids = getattr(output, "prompt_token_ids", None)
                    prompt_item = {
                        "index": count - 1,
                        "id": record["id"],
                        "dataset": record["dataset"],
                        "answer": record["answer"],
                        "pred": pred,
                        "correct": is_correct,
                        "image_path": record["image_path"],
                        "raw_prompt": record["prompt"],
                        "vllm_output_prompt": getattr(output, "prompt", ""),
                        "prompt_token_ids": prompt_token_ids or [],
                        "prompt_token_count": len(prompt_token_ids or []),
                        "decoded_prompt_token_ids": decode_prompt_token_ids(tokenizer, prompt_token_ids),
                        "response": response,
                        "metadata": record.get("metadata", {}),
                    }
                    write_prompt_log(prompt_log, prompt_item)
                    if args.log_prompts_to_console:
                        print_prompt_log(prompt_item)
                out.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "dataset": record["dataset"],
                            "answer": record["answer"],
                            "pred": pred,
                            "correct": is_correct,
                            "response": response,
                            "prompt": record["prompt"],
                            "image_path": record["image_path"],
                            "metadata": record.get("metadata", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
    if prompt_log is not None:
        prompt_log.close()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "count": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "model": args.model,
        "records": str(args.records),
        "output_file": str(output_file),
        "limit": "" if args.limit is None else str(args.limit),
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    update_summary_csv(args.out_dir / "summary.csv", {k: str(summary[k]) for k in SUMMARY_FIELDS})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
