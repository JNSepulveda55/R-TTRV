#!/usr/bin/env python3
"""Plot base-eval accuracy results from summary.csv."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "base-evals-matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


DATASET_LABELS = {
    "imagenet_a": "ImageNet-A",
    "dtd": "DTD",
    "seed": "SEED",
    "ai2d": "AI2D",
}
PAPER_BASE_ACCURACY = {
    "imagenet_a": 67.92,
    "dtd": 37.24,
    "seed": 24.99,
    "ai2d": 39.68,
}
BAR_COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#14B8A6"]
PAPER_LINE_COLOR = "#0F172A"


def read_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def choose_rows(rows: list[dict]) -> list[dict]:
    """Pick one result per dataset, preferring full runs over smoke runs."""
    selected: dict[str, dict] = {}
    for row in rows:
        dataset = row["dataset"]
        current = selected.get(dataset)
        if current is None:
            selected[dataset] = row
            continue
        row_is_full = row.get("limit", "") == ""
        current_is_full = current.get("limit", "") == ""
        if row_is_full and not current_is_full:
            selected[dataset] = row
        elif row_is_full == current_is_full and int(row["count"]) >= int(current["count"]):
            selected[dataset] = row
    order = ["imagenet_a", "dtd", "seed", "ai2d"]
    return sorted(selected.values(), key=lambda row: order.index(row["dataset"]) if row["dataset"] in order else 99)


def plot(rows: list[dict], output: Path, title: str, description: str) -> None:
    datasets = [DATASET_LABELS.get(row["dataset"], row["dataset"]) for row in rows]
    dataset_keys = [row["dataset"] for row in rows]
    accuracies = [float(row["accuracy"]) * 100.0 for row in rows]
    paper_accuracies = [PAPER_BASE_ACCURACY.get(key) for key in dataset_keys]
    counts = [int(row["count"]) for row in rows]
    model = rows[0].get("model", "model") if rows else "model"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    x_positions = list(range(len(datasets)))
    width = 0.64
    bars = ax.bar(
        x_positions,
        accuracies,
        width=width,
        color=BAR_COLORS[: len(datasets)],
        edgecolor="#0F172A",
        linewidth=0.8,
    )

    all_scores = accuracies + [score for score in paper_accuracies if score is not None]
    ax.set_ylim(0, max(100, max(all_scores, default=0) + 12))
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#334155")
    ax.set_title(title, fontsize=17, pad=22, color="#0F172A")
    ax.text(
        0.5,
        1.025,
        description,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(x_positions, datasets)
    ax.tick_params(axis="x", labelsize=11, colors="#0F172A")
    ax.tick_params(axis="y", colors="#475569")

    paper_label_added = False
    for bar, paper_accuracy in zip(bars, paper_accuracies):
        if paper_accuracy is None:
            continue
        x_left = bar.get_x()
        x_right = bar.get_x() + bar.get_width()
        ax.plot(
            [x_left, x_right],
            [paper_accuracy, paper_accuracy],
            color=PAPER_LINE_COLOR,
            linewidth=2.8,
            solid_capstyle="round",
            label="Paper base InternVL3-2B" if not paper_label_added else None,
        )
        paper_label_added = True
        ax.text(
            x_right + 0.04,
            paper_accuracy,
            f"{paper_accuracy:.1f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=PAPER_LINE_COLOR,
        )

    for bar, accuracy, count in zip(bars, accuracies, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(1.5, bar.get_height() * 0.08),
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#DBEAFE",
            fontweight="bold",
        )

    # -- Make a custom nice-looking legend -- #
        this_run_handle = tuple(
        Patch(
            facecolor=color,
            edgecolor="#0F172A",
            linewidth=0.8,
        )
        for color in BAR_COLORS[: len(datasets)]
    )

    paper_handle = Patch(
        facecolor="black",
        edgecolor=PAPER_LINE_COLOR,
        linewidth=2.8,
    )

    handles = [this_run_handle]
    labels = ["My Results"]

    if paper_label_added:
        handles.append(paper_handle)
        labels.append("Paper Results")

    ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    # -- End custom legend -- #


    fig.text(
        0.5,
        0.035,
        f"Model: {model} | Greedy vLLM decoding | ABCD multiple-choice prompts",
        ha="center",
        fontsize=9,
        color="#64748B",
    )
    fig.tight_layout(rect=[0.04, 0.1, 0.98, 0.94])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("base-evals/results/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("base-evals/results/accuracy_bars.png"))
    parser.add_argument("--title", default="Base InternVL3-2B Evaluation Accuracy")
    parser.add_argument(
        "--description",
        default="This run compared with the paper's reported base InternVL3-2B scores, both without TTRV.",
    )
    args = parser.parse_args()

    rows = choose_rows(read_summary(args.summary))
    if not rows:
        raise RuntimeError(f"No rows found in {args.summary}")
    plot(rows, args.output, args.title, args.description)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
