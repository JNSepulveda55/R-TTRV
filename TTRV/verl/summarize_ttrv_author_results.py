#!/usr/bin/env python3
"""Collect TTRV author-code reproduction metrics from Slurm logs."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/-]*)=([^\s]+)")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_./@-]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
STEP_RE = re.compile(r"\bstep:(?P<step>\d+)")

SUMMARY_COLUMNS = [
    "dataset",
    "task",
    "model",
    "train_size",
    "job_id",
    "log_path",
    "initial_val_ttrl_label_accuracy",
    "final_val_ttrl_label_accuracy",
    "initial_val_core_acc",
    "final_val_core_acc",
    "final_train_pass_at_32",
    "final_train_post_pass_at_16",
]


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_author_metadata(line: str) -> dict[str, str]:
    if "AUTHOR_REPRO" not in line:
        return {}
    return {key: value for key, value in KV_RE.findall(line)}


def parse_metric_line(line: str) -> dict[str, float | int]:
    if "step:" not in line:
        return {}
    metrics: dict[str, float | int] = {}
    step_match = STEP_RE.search(line)
    if step_match:
        metrics["step"] = int(step_match.group("step"))
    for match in METRIC_RE.finditer(line):
        key = match.group("key")
        if key == "step":
            continue
        metrics[key] = float(match.group("value"))
    return metrics


def first_metric(metrics: list[dict], key: str) -> float | str:
    for item in metrics:
        if key in item:
            return item[key]
    return ""


def last_metric(metrics: list[dict], key: str) -> float | str:
    for item in reversed(metrics):
        if key in item:
            return item[key]
    return ""


def first_matching(metrics: list[dict], pattern: re.Pattern[str]) -> float | str:
    for item in metrics:
        for key, value in item.items():
            if isinstance(key, str) and pattern.fullmatch(key):
                return value
    return ""


def last_matching(metrics: list[dict], pattern: re.Pattern[str]) -> float | str:
    for item in reversed(metrics):
        for key, value in item.items():
            if isinstance(key, str) and pattern.fullmatch(key):
                return value
    return ""


def parse_log(path: Path) -> tuple[dict[str, str], list[dict]]:
    metadata: dict[str, str] = {}
    metrics = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = clean_line(raw_line)
            metadata.update(parse_author_metadata(line))
            parsed_metrics = parse_metric_line(line)
            if parsed_metrics:
                metrics.append(parsed_metrics)

    return metadata, metrics


def resolved_log_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def summarize_metrics(path: Path, metadata: dict[str, str], metrics: list[dict]) -> dict[str, str | float]:
    val_core_acc = re.compile(r"val-core/.+/acc/mean@\d+")
    train_pass = re.compile(r"train/pass@32")
    train_post_pass = re.compile(r"train/post_pass@16")

    return {
        "dataset": metadata.get("dataset", ""),
        "task": metadata.get("task", ""),
        "model": metadata.get("model", ""),
        "train_size": metadata.get("limit_train") or metadata.get("train_size", ""),
        "job_id": metadata.get("job_id", ""),
        "log_path": str(path),
        "initial_val_ttrl_label_accuracy": first_metric(metrics, "val-ttrl/label_accuracy"),
        "final_val_ttrl_label_accuracy": last_metric(metrics, "val-ttrl/label_accuracy"),
        "initial_val_core_acc": first_matching(metrics, val_core_acc),
        "final_val_core_acc": last_matching(metrics, val_core_acc),
        "final_train_pass_at_32": last_matching(metrics, train_pass),
        "final_train_post_pass_at_16": last_matching(metrics, train_post_pass),
    }


def summarize_logs(paths: list[Path]) -> list[dict[str, str | float]]:
    parsed_by_path: dict[Path, tuple[dict[str, str], list[dict]]] = {}
    metadata_by_run_log: dict[Path, dict[str, str]] = {}

    for path in paths:
        resolved = path.resolve()
        parsed_by_path[resolved] = parse_log(path)

    for resolved, (metadata, _metrics) in parsed_by_path.items():
        if "log_path" in metadata:
            run_log = resolved_log_path(metadata["log_path"], resolved.parent)
            metadata_by_run_log.setdefault(run_log, {}).update(metadata)

    rows = []
    for resolved, (metadata, metrics) in parsed_by_path.items():
        if not metrics:
            continue

        run_log = resolved_log_path(metadata["log_path"], resolved.parent) if "log_path" in metadata else resolved
        if resolved.suffix == ".out" and run_log in parsed_by_path and run_log != resolved:
            continue

        merged_metadata = dict(metadata_by_run_log.get(resolved, {}))
        merged_metadata.update(metadata)
        rows.append(summarize_metrics(Path(merged_metadata.get("log_path", str(resolved))), merged_metadata, metrics))

    return rows


def expand_logs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches if Path(match).is_file())
            continue
        path = Path(pattern)
        if path.is_file():
            paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["logs/author-repro/*.log", "logs/author-repro/*.out"],
        help="Log files or glob patterns to summarize.",
    )
    parser.add_argument("--out", default="results/author-repro/summary.csv")
    args = parser.parse_args()

    log_paths = expand_logs(args.logs)
    if not log_paths:
        raise SystemExit(f"No log files matched: {', '.join(args.logs)}")

    rows = summarize_logs(log_paths)
    if not rows:
        raise SystemExit(f"No metric-bearing log files matched: {', '.join(args.logs)}")
    write_summary(rows, Path(args.out))
    print(f"wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
