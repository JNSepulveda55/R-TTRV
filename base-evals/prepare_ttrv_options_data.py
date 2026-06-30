#!/usr/bin/env python3
"""Build eval JSONL files using TTRV's original ABCD options.

The previous prepare_data.py script saves local HF images and creates fresh
distractors. This script keeps those local image paths, but replaces the prompt
and answer with the prompt/answer from the TTRV test JSONs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable


LETTERS = "ABCD"
DEFAULT_DATASETS = ("imagenet_a", "dtd", "seed", "ai2d")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_dataset_list(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return datasets


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prompt_options(prompt: str) -> list[tuple[str, str]]:
    return re.findall(r"(?m)^\s*([A-D])\.\s+(.+?)\s*$", prompt)


def validate_ttrv_row(dataset: str, idx: int, row: dict) -> None:
    prompt = row.get("prompt")
    answer = str(row.get("answer", "")).strip().upper()
    options = prompt_options(prompt or "")
    option_letters = [letter for letter, _ in options]
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError(f"{dataset} row {idx} has no prompt.")
    if answer not in LETTERS:
        raise RuntimeError(f"{dataset} row {idx} has invalid answer {answer!r}.")
    if option_letters != list(LETTERS):
        raise RuntimeError(
            f"{dataset} row {idx} must contain exactly A/B/C/D options; "
            f"found {option_letters!r}."
        )


def image_path_exists(path_text: str, root: Path) -> bool:
    path = Path(path_text)
    return path.exists() if path.is_absolute() else (root / path).exists()


def exact_key(row: dict) -> tuple[str, str]:
    return clean_text(row["prompt"]), str(row["answer"]).strip().upper()


def class_key_from_ttrv_image(row: dict) -> str:
    image_path = str(row.get("image_path", ""))
    key = Path(image_path).parent.name
    if not key:
        raise RuntimeError(f"Could not parse class directory from {image_path!r}.")
    return key


def build_prompt_queues(base_rows: list[dict]) -> dict[tuple[str, str], deque[dict]]:
    queues: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in base_rows:
        queues[exact_key(row)].append(row)
    return queues


def build_metadata_queues(base_rows: list[dict], metadata_key: str) -> dict[str, deque[dict]]:
    queues: dict[str, deque[dict]] = defaultdict(deque)
    missing = 0
    for row in base_rows:
        value = row.get("metadata", {}).get(metadata_key)
        if value is None:
            missing += 1
            continue
        queues[str(value)].append(row)
    if missing:
        raise RuntimeError(f"{missing} base rows are missing metadata.{metadata_key}.")
    return queues


def compare_bucket_counts(
    dataset: str,
    ttrv_rows: list[dict],
    base_rows: list[dict],
    base_metadata_key: str,
) -> None:
    ttrv_counts = Counter(class_key_from_ttrv_image(row) for row in ttrv_rows)
    base_counts = Counter(str(row.get("metadata", {}).get(base_metadata_key)) for row in base_rows)
    missing = sorted(key for key in ttrv_counts if key not in base_counts)
    deficits = sorted(
        (key, ttrv_counts[key], base_counts[key])
        for key in ttrv_counts
        if base_counts[key] < ttrv_counts[key]
    )
    extras = sorted(key for key in base_counts if key not in ttrv_counts)
    if missing or deficits or extras:
        detail = []
        if missing:
            detail.append(f"missing base buckets: {missing[:8]}")
        if deficits:
            detail.append(f"bucket deficits: {deficits[:8]}")
        if extras:
            detail.append(f"extra base buckets: {extras[:8]}")
        raise RuntimeError(f"{dataset} class bucket mismatch: {'; '.join(detail)}")


def compare_exact_keys(dataset: str, ttrv_rows: list[dict], base_rows: list[dict]) -> None:
    ttrv_counts = Counter(exact_key(row) for row in ttrv_rows)
    base_counts = Counter(exact_key(row) for row in base_rows)
    missing = [key for key, count in ttrv_counts.items() if base_counts[key] < count]
    extras = [key for key, count in base_counts.items() if ttrv_counts[key] < count]
    if missing or extras:
        sample = missing[0] if missing else extras[0]
        prompt, answer = sample
        raise RuntimeError(
            f"{dataset} prompt+answer keys do not match. "
            f"missing={len(missing)}, extras={len(extras)}, "
            f"sample_answer={answer}, sample_prompt={prompt[:180]!r}"
        )


def make_record(
    dataset: str,
    idx: int,
    ttrv_row: dict,
    base_row: dict,
    match_strategy: str,
) -> dict:
    return {
        "id": f"{dataset}-ttrv-{idx:06d}",
        "dataset": dataset,
        "image_path": base_row["image_path"],
        "prompt": ttrv_row["prompt"],
        "answer": str(ttrv_row["answer"]).strip().upper(),
        "metadata": {
            "ttrv_id": ttrv_row.get("id"),
            "ttrv_source": ttrv_row.get("source"),
            "ttrv_original_image_path": ttrv_row.get("image_path"),
            "ttrv_options": [text for _, text in prompt_options(ttrv_row["prompt"])],
            "match_strategy": match_strategy,
            "matched_base_id": base_row.get("id"),
            "matched_base_answer": base_row.get("answer"),
            "matched_base_metadata": base_row.get("metadata", {}),
        },
    }


def convert_exact_prompt_dataset(
    dataset: str,
    ttrv_rows: list[dict],
    base_rows: list[dict],
    limit: int | None,
    root: Path,
) -> list[dict]:
    compare_exact_keys(dataset, ttrv_rows, base_rows)
    queues = build_prompt_queues(base_rows)
    output = []
    selected = ttrv_rows[:limit] if limit else ttrv_rows
    for idx, ttrv_row in enumerate(selected):
        validate_ttrv_row(dataset, idx, ttrv_row)
        key = exact_key(ttrv_row)
        if not queues[key]:
            raise RuntimeError(f"{dataset} row {idx} has no remaining prompt+answer match.")
        base_row = queues[key].popleft()
        if not image_path_exists(str(base_row["image_path"]), root):
            raise RuntimeError(f"Matched image path does not exist: {base_row['image_path']}")
        output.append(make_record(dataset, idx, ttrv_row, base_row, "normalized_prompt_answer"))
    return output


def convert_class_bucket_dataset(
    dataset: str,
    ttrv_rows: list[dict],
    base_rows: list[dict],
    base_metadata_key: str,
    limit: int | None,
    root: Path,
) -> list[dict]:
    compare_bucket_counts(dataset, ttrv_rows, base_rows, base_metadata_key)
    queues = build_metadata_queues(base_rows, base_metadata_key)
    output = []
    selected = ttrv_rows[:limit] if limit else ttrv_rows
    for idx, ttrv_row in enumerate(selected):
        validate_ttrv_row(dataset, idx, ttrv_row)
        key = class_key_from_ttrv_image(ttrv_row)
        if not queues[key]:
            raise RuntimeError(f"{dataset} row {idx} has no remaining local image for class {key!r}.")
        base_row = queues[key].popleft()
        if not image_path_exists(str(base_row["image_path"]), root):
            raise RuntimeError(f"Matched image path does not exist: {base_row['image_path']}")
        output.append(make_record(dataset, idx, ttrv_row, base_row, f"class_bucket:{base_metadata_key}"))
    return output


def convert_dataset(
    dataset: str,
    ttrv_dir: Path,
    base_dir: Path,
    out_dir: Path,
    limit: int | None,
    overwrite: bool,
    root: Path,
) -> int:
    ttrv_path = ttrv_dir / f"{dataset}.json"
    base_path = base_dir / f"{dataset}.jsonl"
    out_path = out_dir / f"{dataset}.jsonl"

    if out_path.exists() and not overwrite:
        print(f"{dataset}: {out_path} exists; pass --overwrite to rebuild.")
        return 0
    if not ttrv_path.exists():
        raise FileNotFoundError(ttrv_path)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    ttrv_rows = json.loads(ttrv_path.read_text(encoding="utf-8"))
    base_rows = read_jsonl(base_path)
    if len(ttrv_rows) != len(base_rows):
        raise RuntimeError(
            f"{dataset} count mismatch: TTRV has {len(ttrv_rows)}, "
            f"base prepared data has {len(base_rows)}."
        )

    for idx, row in enumerate(ttrv_rows):
        validate_ttrv_row(dataset, idx, row)

    if dataset == "imagenet_a":
        records = convert_class_bucket_dataset(
            dataset, ttrv_rows, base_rows, "label_synset", limit, root
        )
    elif dataset == "dtd":
        records = convert_class_bucket_dataset(dataset, ttrv_rows, base_rows, "label_name", limit, root)
    elif dataset in {"seed", "ai2d"}:
        records = convert_exact_prompt_dataset(dataset, ttrv_rows, base_rows, limit, root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    count = write_jsonl(out_path, records)
    expected = min(limit, len(ttrv_rows)) if limit else len(ttrv_rows)
    if count != expected:
        raise RuntimeError(f"{dataset} wrote {count} rows, expected {expected}.")
    print(f"{dataset}: wrote {count} rows to {out_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--ttrv-dir", default="base-evals/data-ttrv-test/ttrv-jsons")
    parser.add_argument("--base-dir", default="base-evals/data")
    parser.add_argument("--out-dir", default="base-evals/data-ttrv-options")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    root = repo_root()
    datasets = parse_dataset_list(args.datasets)
    ttrv_dir = root / args.ttrv_dir
    base_dir = root / args.base_dir
    out_dir = root / args.out_dir

    total = 0
    for dataset in datasets:
        total += convert_dataset(
            dataset=dataset,
            ttrv_dir=ttrv_dir,
            base_dir=base_dir,
            out_dir=out_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            root=root,
        )
    print(f"Done. Wrote {total} rows.")


if __name__ == "__main__":
    main()
