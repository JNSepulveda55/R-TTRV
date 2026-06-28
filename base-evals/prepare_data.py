#!/usr/bin/env python3
"""Prepare simple ABCD eval JSONL files for base VLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm


LETTERS = "ABCD"
DEFAULT_DATASETS = ("imagenet_a", "dtd", "seed", "ai2d")
MAX_IMAGE_DIM = 1000


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def image_too_large(image: Image.Image) -> bool:
    return max(image.size) > MAX_IMAGE_DIM


def first_image(value: object) -> Image.Image | None:
    if isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    if isinstance(value, Image.Image):
        return value
    return None


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=95)


def options_block(options: list[str]) -> str:
    return "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))


def object_prompt(options: list[str]) -> str:
    return (
        "<image> \n Look at the given image and identify what it shows. "
        "Choose the correct answer from the options below and respond with only "
        "the corresponding option letter (A, B, C, or D). Do not include any "
        f"explanation or extra text. \n Options:\n{options_block(options)}"
    )


def texture_prompt(options: list[str]) -> str:
    return (
        "<image> \n Look at the given image and identify what texture it shows. "
        "Choose the correct answer from the options below and respond with only "
        "the corresponding option letter (A, B, C, or D). Do not include any "
        f"explanation or extra text. \n Options:\n{options_block(options)}"
    )


def vqa_prompt(question: str, options: list[str]) -> str:
    return (
        f"<image> \n{clean_text(question)}\n"
        "Choose the correct answer from the options below and respond with only "
        "the corresponding option letter (A, B, C, or D). Do not include any "
        f"explanation or extra text.\nOptions:\n{options_block(options)}"
    )


def make_abcd_options(
    dataset: str,
    example_id: str,
    class_names: list[str],
    correct_name: str,
) -> tuple[list[str], str]:
    candidates = [name for name in class_names if name != correct_name]
    rng = random.Random(stable_seed(0, dataset, example_id, "distractors"))
    options = [correct_name] + rng.sample(candidates, 3)
    random.Random(stable_seed(0, dataset, example_id, "shuffle")).shuffle(options)
    return options, LETTERS[options.index(correct_name)]


def build_imagenet_a_name_map() -> dict[str, str]:
    path = repo_root() / "TTRV/verl/data/imagenet_a_20/test.json"
    rows = json.loads(path.read_text())
    synset_to_name: dict[str, str] = {}
    for row in rows:
        synset = next(
            (part for part in str(row["image_path"]).split("/") if re.fullmatch(r"n\d{8}", part)),
            None,
        )
        options = dict(re.findall(r"([A-D])\.\s*([^\n]+)", row["prompt"]))
        answer = row["answer"]
        if synset and answer in options:
            synset_to_name.setdefault(synset, clean_text(options[answer]))
    if len(synset_to_name) < 200:
        raise RuntimeError(
            f"Expected at least 200 ImageNet-A synset names from {path}, "
            f"found {len(synset_to_name)}."
        )
    return synset_to_name


def class_label_names(dataset) -> list[str]:
    feature = dataset.features["label"]
    if not hasattr(feature, "names"):
        raise RuntimeError("Expected a ClassLabel feature named 'label'.")
    return list(feature.names)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_imagenet_a(out_dir: Path, limit: int | None) -> Iterable[dict]:
    ds = load_dataset("barkermrl/imagenet-a", split="train")
    synset_to_name = build_imagenet_a_name_map()
    synsets = class_label_names(ds)
    class_names = [synset_to_name[synset] for synset in synsets]
    kept = 0
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="imagenet_a"):
        image = first_image(row["image"])
        if image is None or image_too_large(image):
            continue
        label = int(row["label"])
        correct_name = class_names[label]
        record_id = f"imagenet_a-{idx:06d}"
        options, answer = make_abcd_options("imagenet_a", record_id, class_names, correct_name)
        image_path = out_dir / "images/imagenet_a" / f"{record_id}.jpg"
        save_image(image, image_path)
        kept += 1
        yield {
            "id": record_id,
            "dataset": "imagenet_a",
            "image_path": str(image_path),
            "prompt": object_prompt(options),
            "answer": answer,
            "metadata": {
                "source_dataset": "barkermrl/imagenet-a",
                "source_split": "train",
                "source_index": idx,
                "label": label,
                "label_synset": synsets[label],
                "label_name": correct_name,
                "options": options,
                "image_size": list(image.size),
            },
        }
        if limit and kept >= limit:
            break


def prepare_dtd(out_dir: Path, limit: int | None) -> Iterable[dict]:
    train = load_dataset("tanganke/dtd", split="train")
    test = load_dataset("tanganke/dtd", split="test")
    ds = concatenate_datasets([train, test])
    class_names = [clean_text(name) for name in class_label_names(ds)]
    kept = 0
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="dtd"):
        image = first_image(row["image"])
        if image is None or image_too_large(image):
            continue
        label = int(row["label"])
        correct_name = class_names[label]
        record_id = f"dtd-{idx:06d}"
        options, answer = make_abcd_options("dtd", record_id, class_names, correct_name)
        image_path = out_dir / "images/dtd" / f"{record_id}.jpg"
        save_image(image, image_path)
        kept += 1
        yield {
            "id": record_id,
            "dataset": "dtd",
            "image_path": str(image_path),
            "prompt": texture_prompt(options),
            "answer": answer,
            "metadata": {
                "source_dataset": "tanganke/dtd",
                "source_split": "train+test",
                "source_index": idx,
                "label": label,
                "label_name": correct_name,
                "options": options,
                "image_size": list(image.size),
            },
        }
        if limit and kept >= limit:
            break


def prepare_seed(out_dir: Path, limit: int | None) -> Iterable[dict]:
    ds = load_dataset("lmms-lab/SEED-Bench", split="test")
    kept = 0
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="seed"):
        if row.get("data_type") != "image":
            continue
        image = first_image(row["image"])
        if image is None or image_too_large(image):
            continue
        options = [
            clean_text(row["choice_a"]),
            clean_text(row["choice_b"]),
            clean_text(row["choice_c"]),
            clean_text(row["choice_d"]),
        ]
        answer = clean_text(row["answer"]).upper()
        if answer not in LETTERS:
            continue
        record_id = f"seed-{idx:06d}"
        image_path = out_dir / "images/seed" / f"{record_id}.jpg"
        save_image(image, image_path)
        kept += 1
        yield {
            "id": record_id,
            "dataset": "seed",
            "image_path": str(image_path),
            "prompt": vqa_prompt(row["question"], options),
            "answer": answer,
            "metadata": {
                "source_dataset": "lmms-lab/SEED-Bench",
                "source_split": "test",
                "source_index": idx,
                "question_id": row.get("question_id"),
                "data_id": row.get("data_id"),
                "question_type_id": row.get("question_type_id"),
                "options": options,
                "image_size": list(image.size),
            },
        }
        if limit and kept >= limit:
            break


def ai2d_answer_to_letter(answer: object, options: list[str]) -> str | None:
    text = clean_text(answer)
    if text.upper() in LETTERS:
        return text.upper()
    if text.isdigit():
        idx = int(text)
        return LETTERS[idx] if 0 <= idx < 4 else None
    try:
        return LETTERS[options.index(text)]
    except ValueError:
        return None


def prepare_ai2d(out_dir: Path, limit: int | None) -> Iterable[dict]:
    ds = load_dataset("lmms-lab/ai2d", split="test")
    kept = 0
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="ai2d"):
        image = first_image(row["image"])
        if image is None or image_too_large(image):
            continue
        options = [clean_text(option) for option in row["options"]]
        if len(options) != 4:
            continue
        answer = ai2d_answer_to_letter(row["answer"], options)
        if answer is None:
            continue
        record_id = f"ai2d-{idx:06d}"
        image_path = out_dir / "images/ai2d" / f"{record_id}.jpg"
        save_image(image, image_path)
        kept += 1
        yield {
            "id": record_id,
            "dataset": "ai2d",
            "image_path": str(image_path),
            "prompt": vqa_prompt(row["question"], options),
            "answer": answer,
            "metadata": {
                "source_dataset": "lmms-lab/ai2d",
                "source_split": "test",
                "source_index": idx,
                "options": options,
                "image_size": list(image.size),
            },
        }
        if limit and kept >= limit:
            break


PREPARE_FNS = {
    "imagenet_a": prepare_imagenet_a,
    "dtd": prepare_dtd,
    "seed": prepare_seed,
    "ai2d": prepare_ai2d,
}


def parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown datasets: {', '.join(unknown)}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=parse_datasets, default=list(DEFAULT_DATASETS))
    parser.add_argument("--out-dir", type=Path, default=Path("base-evals/data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        out_path = args.out_dir / f"{dataset}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists; pass --overwrite to rebuild.")
            continue
        count = write_jsonl(out_path, PREPARE_FNS[dataset](args.out_dir, args.limit))
        print(f"[done] wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()
