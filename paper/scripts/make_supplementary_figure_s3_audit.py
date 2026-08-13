"""Create Supplementary Figure S3 from the frozen human-240 audit tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports/evaluation/pruscope_human_240_provenance_v1"
SELECTION = (
    PROJECT_ROOT
    / "reports/human_annotation/pruscope_human_240_v1/FIXED_SELECTION.csv"
)
FIGURE_DIR = PROJECT_ROOT / "manuscript/figures"

STAGE_LABELS = {
    "small_green": "Small green",
    "medium_green": "Medium green",
    "mature": "Mature",
}
COLORS = {
    "accepted_unchanged": "#0072B2",
    "edited_prediction": "#E69F00",
    "manual_added": "#6F4E9C",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    provenance = read_csv(REPORT_DIR / "provenance_recall.csv")
    manual = read_csv(REPORT_DIR / "manual_added_stage_size_recall.csv")
    annotator = [
        row
        for row in read_csv(REPORT_DIR / "annotator_subset_audit.csv")
        if row["stage"] == "all"
    ]
    selection = read_csv(SELECTION)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)

    # a: provenance-specific recall across immutable confidence cuts.
    axis = axes[0, 0]
    thresholds = [0.25, 0.05, 0.001]
    x = np.arange(len(thresholds))
    for origin in ("accepted_unchanged", "edited_prediction", "manual_added"):
        values = [
            float(
                next(
                    row["recall_iou50"]
                    for row in provenance
                    if float(row["confidence_threshold"]) == threshold
                    and row["truth_provenance"] == origin
                )
            )
            for threshold in thresholds
        ]
        label = origin.replace("_", " ").capitalize()
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=COLORS[origin],
            label=label,
        )
    axis.set_xticks(x, ["0.25\n(packet)", "0.05", "0.001\n(floor)"])
    axis.set_ylim(-0.03, 1.04)
    axis.set_ylabel("IoU50 recall")
    axis.set_title("Reference-truth provenance")
    axis.legend(frameon=False, loc="lower left")
    panel_label(axis, "a")

    # b: manual additions are the least model-anchored truth.
    axis = axes[0, 1]
    stages = ["small_green", "medium_green", "mature"]
    sizes = ["small", "medium", "large"]
    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["recall_iou50"]
                        for row in manual
                        if float(row["confidence_threshold"]) == 0.001
                        and row["stage"] == stage
                        and row["size"] == size
                    )
                )
                for size in sizes
            ]
            for stage in stages
        ]
    )
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=0.75, aspect="auto")
    axis.set_xticks(range(3), [item.capitalize() for item in sizes])
    axis.set_yticks(range(3), [STAGE_LABELS[item] for item in stages])
    axis.set_title("Manual-added recall at confidence 0.001")
    for row_index in range(3):
        for column_index in range(3):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.46 else "#222222",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
    colorbar.set_label("IoU50 recall", fontsize=8)
    panel_label(axis, "b")

    # c: acquisition-sensitive luminance description, not a causal stage test.
    axis = axes[1, 0]
    luminance = [
        [
            float(row["mean_luminance"])
            for row in selection
            if row["stage_audit_stratum"] == stage
        ]
        for stage in stages
    ]
    boxplot = axis.boxplot(
        luminance,
        tick_labels=[f"{STAGE_LABELS[stage]}\n(n={len(values)})" for stage, values in zip(stages, luminance)],
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={"marker": ".", "markersize": 3, "markerfacecolor": "#666666"},
    )
    for patch, color in zip(boxplot["boxes"], ("#56B4E9", "#E69F00", "#7A6FAC")):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#444444")
    axis.set_ylabel("Full-image mean luminance (0–255)")
    axis.set_title("Acquisition-sensitive image brightness")
    panel_label(axis, "c")

    # d: annotator and subset are inseparable.
    axis = axes[1, 1]
    names = ["Annotator A\n(n=160 images)", "Annotator B\n(n=80 images)"]
    bottoms = np.zeros(2)
    for origin in ("accepted_unchanged", "edited_prediction", "manual_added"):
        values = np.asarray([float(row[origin]) for row in annotator])
        totals = np.asarray([float(row["final_instances"]) for row in annotator])
        fractions = values / totals
        axis.bar(
            range(2),
            fractions,
            bottom=bottoms,
            width=0.62,
            color=COLORS[origin],
            label=origin.replace("_", " ").capitalize(),
        )
        for index, (bottom, fraction, count) in enumerate(zip(bottoms, fractions, values)):
            if fraction >= 0.055:
                axis.text(
                    index,
                    bottom + fraction / 2,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    color="white" if origin != "edited_prediction" else "#222222",
                    fontsize=7.5,
                )
        bottoms += fractions
    axis.set_xticks(range(2), names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of final reference boxes")
    axis.set_title("Annotator–subset edit composition")
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "d")

    figure.suptitle(
        "Human-240 model-assisted reference provenance and acquisition audit",
        fontsize=11,
        fontweight="bold",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figS3_human240_provenance_and_acquisition_audit"
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png").resolve())


if __name__ == "__main__":
    main()
