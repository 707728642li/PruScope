"""Create Fig. 4: DCOH stage discrimination, ablations, and stress tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports" / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
AUDITED_ROOT = REPORT_ROOT / "pruscope_dcoh_human_240_v1"

INK = "#17212B"
MUTED = "#566573"
BLUE = "#0072B2"
SKY = "#56B4E9"
TEAL = "#009E73"
AMBER = "#E69F00"
PURPLE = "#7A5195"
LIGHT_GRAY = "#D8DEE3"
PALE_GRAY = "#F4F6F7"

STAGES = ("small_green", "medium_green", "mature")
STAGE_LABELS = ("Small green", "Medium green", "Mature")
STAGE_COLORS = (TEAL, BLUE, AMBER)


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_paths() -> dict[str, Path]:
    audited = {
        "main_metrics": AUDITED_ROOT / "main_normal" / "metrics.json",
        "visual_metrics": AUDITED_ROOT / "nogeom_normal" / "metrics.json",
        "geometry_metrics": AUDITED_ROOT / "geometry_only_normal" / "metrics.json",
    }
    missing = [str(path) for path in audited.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing finalized human-audit results: {missing}")
    return audited


RESULTS = result_paths()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def plot_confusion(ax: plt.Axes, matrix: np.ndarray) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pruscope_blue", ["#FFFFFF", SKY, BLUE]
    )
    ax.imshow(normalized, vmin=0, vmax=1, cmap=cmap, aspect="equal")
    for row in range(3):
        for col in range(3):
            value = normalized[row, col]
            color = "white" if value > 0.58 else INK
            ax.text(
                col,
                row,
                f"{int(matrix[row, col])}\n{value * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    ax.set_xticks(range(3), STAGE_LABELS, rotation=28, ha="right")
    ax.set_yticks(range(3), STAGE_LABELS)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Reference stage")
    ax.set_title("Image-level confusion matrix", fontsize=9, pad=7, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_developmental_index() -> dict[str, list[float]]:
    path = AUDITED_ROOT / "main_normal" / "image_predictions.csv"
    values = {stage: [] for stage in STAGES}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["target"]].append(float(row["mean_developmental_index"]))
    return values


def plot_developmental_index(ax: plt.Axes, values: dict[str, list[float]]) -> None:
    arrays = [np.asarray(values[stage], dtype=float) for stage in STAGES]
    violins = ax.violinplot(
        arrays,
        positions=np.arange(3),
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], STAGE_COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(INK)
        body.set_linewidth(0.65)
        body.set_alpha(0.30)

    rng = np.random.default_rng(20260805)
    for position, (array, color) in enumerate(zip(arrays, STAGE_COLORS)):
        jitter = rng.uniform(-0.20, 0.20, size=len(array))
        ax.scatter(
            np.full_like(array, position) + jitter,
            array,
            s=8,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.63,
            zorder=3,
        )
        q1, median, q3 = np.quantile(array, [0.25, 0.50, 0.75])
        ax.plot([position, position], [q1, q3], color=INK, lw=3.2, zorder=4)
        ax.scatter([position], [median], s=20, facecolor="white", edgecolor=INK, zorder=5)

    stage_tick_labels = [
        f"{label}\n(n={len(array)})" for label, array in zip(STAGE_LABELS, arrays)
    ]
    ax.set_xticks(range(3), stage_tick_labels, rotation=18, ha="right")
    ax.set_ylabel("Ordered cross-cohort index (0–2)")
    ax.set_ylim(-0.15, 2.15)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_title(
        "Ordered cross-cohort index by reference stage",
        fontsize=9,
        pad=7,
        color=INK,
    )
    style_axis(ax)


def metric_with_ci(payload: dict, metric: str) -> tuple[float, float, float]:
    estimate = float(payload["image_metrics"][metric])
    interval = payload["image_bootstrap_95_ci"][metric]
    return estimate, float(interval["lower_95"]), float(interval["upper_95"])


def plot_ablation(ax: plt.Axes) -> None:
    geometry = read_json(RESULTS["geometry_metrics"])
    visual = read_json(RESULTS["visual_metrics"])
    combined = read_json(RESULTS["main_metrics"])
    variants = ("Geometry only", "Visual only", "Visual + geometry")
    payloads = (geometry, visual, combined)
    colors = (MUTED, SKY, PURPLE)
    positions = np.arange(3)

    for offset, metric, marker, label in (
        (-0.055, "macro_f1", "o", "Macro F1"),
        (0.055, "quadratic_weighted_kappa", "s", "Quadratic weighted $\kappa$"),
    ):
        estimates, lower, upper = zip(*(metric_with_ci(p, metric) for p in payloads))
        estimates = np.asarray(estimates)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        for x, y, lo, hi, color in zip(positions + offset, estimates, lower, upper, colors):
            ax.errorbar(
                x,
                y,
                yerr=[[y - lo], [hi - y]],
                fmt=marker,
                ms=5.4,
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.45,
                ecolor=color,
                capsize=2.5,
                lw=1.15,
                zorder=3,
            )
        ax.plot([], [], marker=marker, color=INK, linestyle="none", label=label)

    ax.set_xticks(positions, variants, rotation=18, ha="right")
    ax.set_ylabel("Image-level score")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title("Modality ablation (95% bootstrap CI)", fontsize=9, pad=7, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", handletextpad=0.4)
    style_axis(ax)


def plot_robustness(ax: plt.Axes) -> None:
    conditions = ("normal", "center_only", "background_only", "grayscale")
    labels = ("Normal", "Fruit center", "Background", "Grayscale")
    x = np.arange(len(conditions))

    run_names = {
        "main": {
            "normal": "main_normal",
            "center_only": "main_center_only",
            "background_only": "main_background_only",
            "grayscale": "main_grayscale",
        },
        "visual_only": {
            "normal": "nogeom_normal",
            "center_only": "visual_center_only",
            "background_only": "visual_background_only",
            "grayscale": "visual_grayscale",
        },
    }

    for system, color, marker, label in (
        ("main", PURPLE, "o", "Visual + geometry"),
        ("visual_only", BLUE, "s", "Visual only"),
    ):
        payloads = [
            read_json(AUDITED_ROOT / run_names[system][condition] / "metrics.json")
            for condition in conditions
        ]
        f1 = [payload["image_metrics"]["macro_f1"] for payload in payloads]
        kappa = [payload["image_metrics"]["quadratic_weighted_kappa"] for payload in payloads]
        ax.plot(
            x,
            f1,
            color=color,
            marker=marker,
            ms=5,
            lw=1.5,
            label=f"{label}: macro F1",
        )
        ax.plot(
            x,
            kappa,
            color=color,
            marker=marker,
            ms=4.5,
            lw=1.15,
            linestyle="--",
            markerfacecolor="white",
            label=f"{label}: $\kappa$",
        )

    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Image-level score")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title("Counterfactual image stress tests", fontsize=9, pad=7, color=INK)
    ax.legend(frameon=False, fontsize=7.0, loc="lower left", handlelength=2.6)
    style_axis(ax)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal = read_json(RESULTS["main_metrics"])
    confusion = np.asarray(normal["image_metrics"]["confusion_matrix"], dtype=int)

    figure, axes = plt.subplots(2, 2, figsize=(7.25, 6.45), constrained_layout=False)
    plot_confusion(axes[0, 0], confusion)
    plot_developmental_index(axes[0, 1], load_developmental_index())
    plot_ablation(axes[1, 0])
    plot_robustness(axes[1, 1])

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        add_panel_label(ax, label)

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.965, wspace=0.35, hspace=0.48)
    stem = OUTPUT_DIR / "fig4_dcoh_stage_discrimination_and_robustness"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
