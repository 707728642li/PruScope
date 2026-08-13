"""Create Fig. 5 from the validation-locked, human-enhanced final A2 detector.

Every panel uses the same checkpoint lineage and its validation-selected count
threshold. Raw image-level observations remain visible in the statistical
panels; density-bin summaries are descriptive medians and interquartile ranges.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
PREDICTIONS = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "pruscope_postreview_e30_human_external_plum"
    / "human_a2_predictions.jsonl"
)
PLOS_ERRORS = (
    PROJECT_ROOT
    / "reports"
    / "error_analysis"
    / "pruscope_human_a2_external_plum"
    / "image_error_metrics.csv"
)
INTERNAL_ERRORS = (
    PROJECT_ROOT
    / "reports"
    / "error_analysis"
    / "pruscope_human_a2_internal"
    / "image_error_metrics.csv"
)
COUNT_CONFIDENCE = 0.50

INK = "#17212B"
MUTED = "#566573"
BLUE = "#0072B2"
AMBER = "#E69F00"
PURPLE = "#7A5195"
LIGHT_GRID = "#D8DEE3"

EXEMPLARS = (
    ("plos2023__IMG_534.jpg", "High accuracy"),
    ("plos2023__IMG_370.jpg", "Dense success"),
    ("plos2023__IMG_466.jpg", "Dense challenge"),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_prediction_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        if line:
            record = json.loads(line)
            index[Path(record["image_path"]).name] = record
    return index


def read_gt_boxes(image_path: Path, width: int, height: int) -> np.ndarray:
    parts = list(image_path.parts)
    image_index = parts.index("images")
    parts[image_index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    boxes = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, xc, yc, bw, bh = map(float, fields[:5])
        boxes.append(
            [
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[:, 0] * wh[:, 1]
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    union = box_area + areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def match_predictions(
    gt_boxes: np.ndarray,
    predictions: list[dict],
    confidence: float = COUNT_CONFIDENCE,
    iou_threshold: float = 0.50,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    retained = [item for item in predictions if float(item["confidence"]) >= confidence]
    retained.sort(key=lambda item: float(item["confidence"]), reverse=True)
    unmatched_gt = set(range(len(gt_boxes)))
    true_positive, false_positive = [], []
    for item in retained:
        box = np.asarray(item["xyxy"], dtype=float)
        candidates = np.asarray(sorted(unmatched_gt), dtype=int)
        if len(candidates):
            overlaps = box_iou(box, gt_boxes[candidates])
            best_local = int(np.argmax(overlaps))
            if overlaps[best_local] >= iou_threshold:
                unmatched_gt.remove(int(candidates[best_local]))
                true_positive.append(box)
                continue
        false_positive.append(box)
    false_negative = [gt_boxes[index] for index in sorted(unmatched_gt)]
    return true_positive, false_positive, false_negative


def draw_box(ax: plt.Axes, box: np.ndarray, color: str, linestyle: str, width: float) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=width,
            linestyle=linestyle,
        )
    )


def add_photo_panel(ax: plt.Axes, record: dict, heading: str, panel_label: str) -> None:
    path = Path(record["image_path"])
    image = mpimg.imread(path)
    height, width = image.shape[:2]
    gt = read_gt_boxes(path, width, height)
    tp, fp, fn = match_predictions(gt, record["predictions"])
    ax.imshow(image)
    for box in tp:
        draw_box(ax, box, BLUE, "-", 0.85)
    for box in fp:
        draw_box(ax, box, PURPLE, "--", 0.90)
    for box in fn:
        draw_box(ax, box, AMBER, "-", 0.95)
    prediction_count = len(tp) + len(fp)
    f1 = 2 * len(tp) / max(2 * len(tp) + len(fp) + len(fn), 1)
    ax.text(
        0.5,
        -0.055,
        f"{heading}\nRef {len(gt)}  ·  Pred {prediction_count}  ·  F1 {f1:.2f}",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=INK,
        ha="center",
        va="top",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.10",
            "facecolor": "#F2F5F7",
            "edgecolor": LIGHT_GRID,
            "linewidth": 0.55,
        },
    )
    ax.text(
        -0.04,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.set_axis_off()


def read_error_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            rows.append(
                {
                    "targets": float(item["targets"]),
                    "predictions": float(item["predictions"]),
                    "error": float(item["count_error"]),
                }
            )
    return rows


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )


def agreement_metrics(rows: list[dict]) -> tuple[float, float]:
    reference = np.asarray([row["targets"] for row in rows], dtype=float)
    predicted = np.asarray([row["predictions"] for row in rows], dtype=float)
    residual = float(np.square(predicted - reference).sum())
    total = float(np.square(reference - reference.mean()).sum())
    mae = float(np.abs(predicted - reference).mean())
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return mae, r2


def plot_counts(ax: plt.Axes, internal: list[dict], external: list[dict]) -> None:
    maximum = max(row["targets"] for row in internal + external) * 1.04
    ax.plot([0, maximum], [0, maximum], color=MUTED, lw=1.05, linestyle="--", label="Identity")
    for rows, color, marker, label in (
        (internal, BLUE, "o", "Internal"),
        (external, PURPLE, "s", "External plum"),
    ):
        ax.scatter(
            [row["targets"] for row in rows],
            [row["predictions"] for row in rows],
            s=18,
            color=color,
            marker=marker,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    ax.set_xscale("symlog", linthresh=5, linscale=0.8)
    ax.set_yscale("symlog", linthresh=5, linscale=0.8)
    ticks = [0, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_yticks(ticks, [str(value) for value in ticks])
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference fruit count")
    ax.set_ylabel("Predicted fruit count")
    ax.set_title("Image-level count agreement", fontsize=9, color=INK, pad=7)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", handletextpad=0.4)
    internal_mae, internal_r2 = agreement_metrics(internal)
    external_mae, external_r2 = agreement_metrics(external)
    ax.text(
        0.98,
        0.035,
        f"Internal: MAE {internal_mae:.2f}, R² {internal_r2:.3f}\n"
        f"External: MAE {external_mae:.2f}, R² {external_r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=INK,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": LIGHT_GRID,
            "alpha": 0.90,
        },
    )
    style_axis(ax)


def density_group(targets: float) -> int:
    if targets <= 10:
        return 0
    if targets <= 25:
        return 1
    if targets <= 50:
        return 2
    return 3


def plot_residuals(ax: plt.Axes, internal: list[dict], external: list[dict]) -> None:
    rng = np.random.default_rng(20260805)
    ax.axhline(0, color=MUTED, lw=1.0, linestyle="--", zorder=1)
    centers = np.arange(4, dtype=float)
    for rows, color, marker, label, offset in (
        (internal, BLUE, "o", "Internal", -0.16),
        (external, PURPLE, "s", "External plum", 0.16),
    ):
        for group in range(4):
            values = np.asarray(
                [row["error"] for row in rows if density_group(row["targets"]) == group],
                dtype=float,
            )
            if not len(values):
                continue
            jitter = rng.normal(0.0, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), centers[group] + offset) + jitter,
                values,
                s=12,
                color=color,
                marker=marker,
                alpha=0.24,
                edgecolor="none",
                zorder=2,
            )
            median = float(np.median(values))
            q1, q3 = np.quantile(values, [0.25, 0.75])
            ax.errorbar(
                centers[group] + offset,
                median,
                yerr=[[median - q1], [q3 - median]],
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=5.5,
                elinewidth=2.0,
                capsize=3,
                zorder=4,
                label=label if group == 0 else None,
            )
    ax.set_xticks(centers, ["1–10", "11–25", "26–50", ">50"])
    ax.set_xlabel("Reference fruit per image")
    ax.set_ylabel("Count error (predicted − reference)")
    ax.set_title("Error by reference-density stratum", fontsize=9, color=INK, pad=7)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", ncol=2, handletextpad=0.3)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.55, alpha=0.75)
    style_axis(ax)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_index = read_prediction_index()
    internal = read_error_rows(INTERNAL_ERRORS)
    external = read_error_rows(PLOS_ERRORS)

    figure = plt.figure(figsize=(7.25, 5.35))
    grid = figure.add_gridspec(2, 6, height_ratios=(0.67, 1.0), hspace=0.56, wspace=0.50)
    photo_axes = [figure.add_subplot(grid[0, index * 2 : index * 2 + 2]) for index in range(3)]
    for ax, (filename, heading), label in zip(
        photo_axes, EXEMPLARS, ("(a)", "(b)", "(c)"), strict=True
    ):
        add_photo_panel(ax, prediction_index[filename], heading, label)

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.3, label="Matched prediction"),
        Line2D([0], [0], color=PURPLE, lw=1.3, linestyle="--", label="False positive"),
        Line2D([0], [0], color=AMBER, lw=1.3, label="Missed reference"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.575),
        ncol=3,
        fontsize=7.0,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )

    count_axis = figure.add_subplot(grid[1, :3])
    residual_axis = figure.add_subplot(grid[1, 3:])
    plot_counts(count_axis, internal, external)
    plot_residuals(residual_axis, internal, external)
    add_panel_label(count_axis, "(d)")
    add_panel_label(residual_axis, "(e)")

    figure.text(
        0.50,
        0.012,
        f"Final human-enhanced A2 · validation-selected conf = {COUNT_CONFIDENCE:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.965)
    stem = OUTPUT_DIR / "fig5_qualitative_detection_and_count_errors"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
