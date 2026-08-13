"""Create Supplementary Figure S4 for the frozen post-review human-enhancement study.

The script deliberately reads only completed, protocol-locked artifacts.  It does
not select a checkpoint or inspect a protected test before the deployment lock.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports/evaluation/pruscope_postreview_convergence_human_summary"
    / "POSTREVIEW_CONVERGENCE_HUMAN_SUMMARY.json"
)
FIGURE_DIR = PROJECT_ROOT / "manuscript/figures"

INK = "#17212B"
MUTED = "#66717E"
BLUE = "#0072B2"
TEAL = "#009E73"
AMBER = "#E69F00"
PURPLE = "#7A5195"
GRAY = "#D8DEE3"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        f"({label})",
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        color=INK,
    )


def metric_lookup(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (str(row["domain"]), str(row["model"]), str(row["size"])): row
        for row in rows
    }


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(float(row["epoch"])) + 1 for row in rows]),
        np.asarray([float(row["metrics/mAP50-95(B)"]) for row in rows]),
    )


def bootstrap_ap_rows(summary: dict) -> list[tuple[str, str, dict]]:
    rows: list[tuple[str, str, dict]] = []
    for domain in ("internal", "external_plum", "citdet"):
        for size in ("all", "small"):
            match = next(
                row
                for row in summary["paired_human_minus_e30_bootstrap"][domain]
                if row["metric"] == "AP50_95" and row["size"] == size
            )
            rows.append((domain, size, match))
    return rows


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    if summary["status"] != "POSTREVIEW_FIXED_PIPELINE_COMPLETE":
        raise ValueError("Post-review study is incomplete")
    selection = summary["selection_lock"]
    if selection["status"] != "LOCKED_BEFORE_PROTECTED_TEST_EVALUATION":
        raise ValueError("Difficult-target checkpoint was not prospectively locked")

    selected_key = summary["selected_model_key"]
    reference_key = summary["reference_model_key"]
    selected_run = Path(summary["training_runs"][selected_key]["best_weight"]).parents[1]
    reference_run = Path(summary["training_runs"][reference_key]["best_weight"]).parents[1]
    metrics = metric_lookup(summary["size_stratified_metrics"])
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)

    # a: optimization traces, descriptive convergence sensitivity only.
    axis = axes[0, 0]
    for run, label, color in (
        (reference_run, "30-epoch reference", MUTED),
        (selected_run, "Difficult-target update", TEAL),
    ):
        epochs, values = read_curve(run / "results.csv")
        axis.plot(epochs, values, color=color, linewidth=1.6, label=label)
        best = int(np.argmax(values))
        axis.scatter(epochs[best], values[best], color=color, s=18, zorder=3)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Training-log validation AP50–95")
    axis.set_title("Convergence sensitivity")
    axis.grid(False)
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "a")

    # b: protected-domain descriptive AP after the validation-only selection lock.
    axis = axes[0, 1]
    groups = [
        ("internal", "all", "Internal\nall"),
        ("internal", "small", "Internal\nsmall"),
        ("external_plum", "all", "Plum\nall"),
        ("external_plum", "small", "Plum\nsmall"),
        ("citdet", "all", "CitDet\nall"),
        ("citdet", "small", "CitDet\nsmall"),
    ]
    x = np.arange(len(groups))
    width = 0.36
    reference = [float(metrics[(d, reference_key, s)]["AP50_95"]) for d, s, _ in groups]
    selected = [float(metrics[(d, selected_key, s)]["AP50_95"]) for d, s, _ in groups]
    axis.bar(x - width / 2, reference, width, color=MUTED, label="30-epoch reference")
    axis.bar(x + width / 2, selected, width, color=TEAL, label="Difficult-target update")
    axis.set_xticks(x, [label for _, _, label in groups])
    axis.set_ylabel("AP50–95")
    axis.set_title("Post-lock descriptive evaluation")
    axis.grid(False)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "b")

    # c: image-bootstrap uncertainty for the updated-minus-reference contrast.
    axis = axes[1, 0]
    bootstrap = bootstrap_ap_rows(summary)
    labels = [
        f"{ {'internal':'Internal','external_plum':'Plum','citdet':'CitDet'}[domain] }\n{size}"
        for domain, size, _ in bootstrap
    ]
    differences = np.asarray([float(row["difference"]) for _, _, row in bootstrap])
    lower = np.asarray([float(row["lower_95"]) for _, _, row in bootstrap])
    upper = np.asarray([float(row["upper_95"]) for _, _, row in bootstrap])
    y = np.arange(len(bootstrap))
    colors = [
        AMBER if low <= 0 <= high else (TEAL if value > 0 else PURPLE)
        for value, low, high in zip(differences, lower, upper, strict=True)
    ]
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.errorbar(
        differences,
        y,
        xerr=np.vstack([differences - lower, upper - differences]),
        fmt="none",
        ecolor=MUTED,
        elinewidth=1.1,
        capsize=2.5,
        zorder=1,
    )
    axis.scatter(differences, y, color=colors, s=26, zorder=2)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Updated minus reference AP50–95")
    axis.set_title("Paired image-bootstrap differences (95% CI)")
    axis.grid(False)
    panel_label(axis, "c")

    # d: validation-only architecture selection. This deliberately keeps the
    # figure focused on model evidence rather than annotation/image counts.
    axis = axes[1, 1]
    candidates = ["A2", "A5"]
    overall = np.asarray(
        [
            float(summary["selection_lock"]["a2_validation_overall_AP50_95"]),
            float(summary["selection_lock"]["a5_validation_overall_AP50_95"]),
        ]
    )
    small = np.asarray(
        [
            float(summary["selection_lock"]["a2_validation_small_AP50_95"]),
            float(summary["selection_lock"]["a5_validation_small_AP50_95"]),
        ]
    )
    x = np.arange(len(candidates))
    bars_overall = axis.bar(x - width / 2, overall, width, color=MUTED, label="Overall AP50–95")
    bars_small = axis.bar(x + width / 2, small, width, color=TEAL, label="COCO-small AP50–95")
    for bars in (bars_overall, bars_small):
        for bar in bars:
            value = float(bar.get_height())
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
            )
    axis.set_xticks(x, candidates)
    axis.set_ylim(0, 0.66)
    axis.set_ylabel("Validation AP50–95")
    axis.set_title("Validation-only architecture selection", pad=28)
    axis.grid(False)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.2,
        handlelength=1.2,
        columnspacing=0.9,
    )
    panel_label(axis, "d")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figS3_postreview_difficult_target_update"
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png").resolve())


if __name__ == "__main__":
    main()
