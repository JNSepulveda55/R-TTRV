#!/usr/bin/env python3
"""Patch TTRV author JSON image paths and build veRL parquet files.

The author JSON files remain the source of truth for prompts, answers, IDs, and
row order. This script only replaces each row's original image path with the
local image downloaded for the prior base-evals runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import datasets


DEFAULT_DATASETS = ("imagenet_a", "dtd", "seed", "ai2d")
FALLBACK_IMAGE_ONLY = {("imagenet_a", "train"), ("dtd", "train")}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verl_root() -> Path:
    return Path(__file__).resolve().parent


def parse_dataset_list(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("No datasets selected.")
    return names


def read_json_array(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected {path} to contain a JSON array.")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_array(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(rows)


def resolve_local_image(path_text: str, root: Path) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Matched local image does not exist: {path}")
    return str(path)


def build_match_indexes(rows: list[dict], root: Path):
    by_exact = defaultdict(list)
    by_original = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        original_path = str(metadata.get("ttrv_original_image_path", ""))
        answer = str(row.get("answer", "")).strip().upper()
        local_image = resolve_local_image(str(row["image_path"]), root)
        entry = {
            "original_path": original_path,
            "prompt": row.get("prompt"),
            "answer": answer,
            "local_image": local_image,
            "record_id": row.get("id"),
        }
        by_exact[(original_path, row.get("prompt"), answer)].append(entry)
        by_original[original_path].append(entry)
    return by_exact, by_original


def pick_match(dataset: str, split: str, row: dict, by_exact, by_original) -> tuple[str, str]:
    original_path = str(row.get("image_path", ""))
    answer = str(row.get("answer", "")).strip().upper()
    exact_key = (original_path, row.get("prompt"), answer)
    exact_matches = by_exact.get(exact_key, [])
    if len(exact_matches) == 1:
        return exact_matches[0]["local_image"], "exact"
    if len(exact_matches) > 1:
        raise RuntimeError(f"{dataset}/{split} row id={row.get('id')} has ambiguous exact image matches.")

    if (dataset, split) not in FALLBACK_IMAGE_ONLY:
        raise RuntimeError(
            f"{dataset}/{split} row id={row.get('id')} has no exact match for original image, prompt, and answer."
        )

    original_matches = by_original.get(original_path, [])
    if len(original_matches) == 1:
        return original_matches[0]["local_image"], "original_image"
    if not original_matches:
        raise RuntimeError(f"{dataset}/{split} row id={row.get('id')} has no local image for {original_path!r}.")
    raise RuntimeError(f"{dataset}/{split} row id={row.get('id')} has ambiguous original-image matches.")


def patch_rows(
    dataset: str,
    split: str,
    rows: list[dict],
    by_exact,
    by_original,
    limit: int | None,
) -> tuple[list[dict], dict]:
    selected = rows[:limit] if limit else rows
    patched = []
    match_counts = defaultdict(int)
    for row in selected:
        local_image, strategy = pick_match(dataset, split, row, by_exact, by_original)
        new_row = dict(row)
        new_row["image_path"] = local_image
        patched.append(new_row)
        match_counts[strategy] += 1
    return patched, dict(match_counts)


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        images = [{"image": example["image_path"]}]
        return {
            "data_source": "GPQA-TTT",
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
                "author_id": str(example.get("id", "")),
                "author_source": str(example.get("source", "")),
                "dataset": data_source,
            },
            "images": images,
        }

    return process_fn


def write_parquet(json_path: Path, parquet_path: Path, split: str, data_source: str) -> int:
    dataset = datasets.Dataset.from_list(read_json_array(json_path))
    dataset = dataset.map(function=make_map_fn(split, data_source), with_indices=True)
    dataset.to_parquet(str(parquet_path))
    return len(dataset)


def convert_dataset(dataset: str, args, root: Path, verl_dir: Path) -> dict:
    source_dir = Path(args.author_data_dir)
    if not source_dir.is_absolute():
        source_dir = verl_dir / source_dir
    source_dir = source_dir / f"{dataset}_{args.shot}"

    options_dir = Path(args.local_options_dir)
    if not options_dir.is_absolute():
        options_dir = root / options_dir
    options_path = options_dir / f"{dataset}.jsonl"

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = verl_dir / output_root
    output_name = f"{dataset}_{args.shot}{args.name_suffix}"
    output_dir = output_root / output_name

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{output_dir} already exists. Pass --overwrite to rebuild it.")

    local_rows = read_jsonl(options_path)
    by_exact, by_original = build_match_indexes(local_rows, root)

    manifest = {
        "dataset": dataset,
        "source_dir": str(source_dir),
        "options_path": str(options_path),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split, limit in [("train", args.limit_train), ("test", args.limit_test)]:
        source_json = source_dir / f"{split}.json"
        rows = read_json_array(source_json)
        patched_rows, match_counts = patch_rows(dataset, split, rows, by_exact, by_original, limit)
        out_json = output_dir / f"{split}.json"
        out_parquet = output_dir / f"{split}.parquet"
        count = write_json_array(out_json, patched_rows)
        parquet_count = write_parquet(out_json, out_parquet, split, output_name)
        if count != parquet_count:
            raise RuntimeError(f"{dataset}/{split} parquet count {parquet_count} != JSON count {count}.")
        manifest["splits"][split] = {
            "source_rows": len(rows),
            "output_rows": count,
            "limit": limit,
            "match_counts": match_counts,
            "json": str(out_json),
            "parquet": str(out_parquet),
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--shot", type=int, default=20)
    parser.add_argument("--author-data-dir", default="data")
    parser.add_argument("--local-options-dir", default="base-evals/data-ttrv-options")
    parser.add_argument("--output-root", default="data-author-repro")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.limit_train or args.limit_test) and not args.name_suffix:
        raise ValueError("Use --name-suffix for limited smoke datasets so full outputs are not overwritten.")
    for value_name in ["limit_train", "limit_test"]:
        value = getattr(args, value_name)
        if value is not None and value <= 0:
            raise ValueError(f"--{value_name.replace('_', '-')} must be positive.")

    root = repo_root()
    verl_dir = verl_root()
    manifests = []
    for dataset in parse_dataset_list(args.datasets):
        manifest = convert_dataset(dataset, args, root, verl_dir)
        manifests.append(manifest)
        train_rows = manifest["splits"]["train"]["output_rows"]
        test_rows = manifest["splits"]["test"]["output_rows"]
        print(f"{dataset}: wrote train={train_rows}, test={test_rows} to {manifest['output_dir']}")

    print(json.dumps({"converted": manifests}, indent=2))


if __name__ == "__main__":
    main()
