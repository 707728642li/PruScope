"""Create Fig. 6: same-budget detector benchmark and end-to-end stage evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "manuscript" / "final_metrics_registry.json"
OUTPUT = ROOT / "manuscript" / "figures" / "fig6_competitive_benchmark_and_end_to_end_stage"

INK = "#17212B"
MUTED = "#566573"
BLUE = "#0072B2"
TEAL = "#009E73"
AMBER = "#E69F00"
PURPLE = "#7A5195"
GRAY = "#D8DEE3"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.13, 1.08, f"({text})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def main() -> None:
    configure()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    benchmark = registry.get("competitive_detector_benchmark")
    if not benchmark:
        raise RuntimeError("Competitive detector benchmark is not yet bound to the registry")
    end_to_end = registry["end_to_end_stage"]
    systems = benchmark["systems"]
    names = [row["label"] for row in systems]
    colors = [MUTED, PURPLE, TEAL]
    x = np.arange(len(names))
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)

    for ax, domain, title in (
        (axes[0, 0], "internal", "Internal plum test"),
        (axes[0, 1], "external", "External plum test"),
    ):
        width = 0.35
        all_values = [row[f"{domain}_all_AP50_95"] for row in systems]
        small_values = [row[f"{domain}_small_AP50_95"] for row in systems]
        all_sd = [row[f"{domain}_all_AP50_95_sd"] for row in systems]
        small_sd = [row[f"{domain}_small_AP50_95_sd"] for row in systems]
        ax.bar(x - width / 2, all_values, width, color=colors, alpha=0.88, label="Overall")
        ax.bar(x + width / 2, small_values, width, color=colors, alpha=0.45, hatch="///", edgecolor=INK, label="Small")
        ax.errorbar(x - width / 2, all_values, yerr=all_sd, fmt="none", ecolor=INK, capsize=2.5, linewidth=0.9, zorder=3)
        ax.errorbar(x + width / 2, small_values, yerr=small_sd, fmt="none", ecolor=INK, capsize=2.5, linewidth=0.9, zorder=3)
        for index, row in enumerate(systems):
            all_seed = list(row["per_seed"][f"{domain}_all_AP50_95"].values())
            small_seed = list(row["per_seed"][f"{domain}_small_AP50_95"].values())
            jitter = np.asarray((-0.045, 0.0, 0.045))
            ax.scatter(index - width / 2 + jitter, all_seed, s=7, color=INK, alpha=0.75, zorder=4)
            ax.scatter(index + width / 2 + jitter, small_seed, s=7, color=INK, alpha=0.75, zorder=4)
        for xi, value, sd in zip(x - width / 2, all_values, all_sd, strict=True):
            ax.text(xi, value + sd + 0.014, f"{value:.3f}", ha="center", va="bottom", fontsize=6.7)
        for xi, value, sd in zip(x + width / 2, small_values, small_sd, strict=True):
            ax.text(xi, value + sd + 0.014, f"{value:.3f}", ha="center", va="bottom", fontsize=6.7)
        ax.set_xticks(x, names)
        ax.set_ylabel("AP50–95")
        ax.set_ylim(0.15, max(all_values + small_values) + 0.12)
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    metrics = end_to_end["fruit_bearing_metrics"]
    boot = end_to_end["fruit_bearing_sensitivity"]["results"]
    stages = ["Small green", "Medium green", "Mature"]
    matrix = np.asarray(metrics["image_confusion_with_no_detection"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image_ax = axes[1, 0]
    heat = image_ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            image_ax.text(col, row, f"{int(matrix[row,col])}\n{normalized[row,col]:.1%}", ha="center", va="center", fontsize=7, color="white" if normalized[row,col] > 0.55 else INK)
    image_ax.set_xticks(range(4), stages + ["No detection"], rotation=25, ha="right")
    image_ax.set_yticks(range(3), stages)
    image_ax.set_xlabel("End-to-end prediction")
    image_ax.set_ylabel("Reference stage")
    image_ax.set_title("Image-level pipeline outcome", fontsize=9)
    figure.colorbar(heat, ax=image_ax, fraction=0.046, pad=0.04, label="Row proportion")

    summary_ax = axes[1, 1]
    entries = [
        ("Image coverage", "image_coverage", TEAL),
        ("Image accuracy\n(no detection = error)", "image_accuracy_with_no_detection_as_error", BLUE),
        ("Image macro F1\n(no detection = error)", "image_macro_f1_with_no_detection_as_error", PURPLE),
        ("Joint correct-stage\nfruit recall", "joint_correct_stage_recall_over_all_gt", AMBER),
    ]
    positions = np.arange(len(entries))
    estimates = [boot[key]["estimate"] for _, key, _ in entries]
    lower = [estimate - boot[key]["lower_95"] for estimate, (_, key, _) in zip(estimates, entries, strict=True)]
    upper = [boot[key]["upper_95"] - estimate for estimate, (_, key, _) in zip(estimates, entries, strict=True)]
    summary_ax.barh(positions, estimates, color=[color for _, _, color in entries], alpha=0.9, zorder=2)
    summary_ax.errorbar(estimates, positions, xerr=np.asarray([lower, upper]), fmt="none", ecolor=INK, capsize=3, linewidth=1, zorder=3)
    for position, estimate, margin in zip(positions, estimates, upper, strict=True):
        summary_ax.text(estimate + margin + 0.025, position, f"{estimate:.3f}", va="center", fontsize=7)
    summary_ax.set_yticks(positions, [name for name, _, _ in entries], fontsize=7)
    summary_ax.invert_yaxis()
    summary_ax.set_xlim(0, 1.08)
    summary_ax.set_xlabel("Proportion")
    summary_ax.set_title("Full-pipeline metrics (95% image bootstrap CI)", fontsize=9)
    style(summary_ax)

    for ax, panel in zip(axes.flat, ("a", "b", "c", "d"), strict=True):
        label(ax, panel)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
