"""Create Supplementary Fig. S1: post-audit MARS optimization and confirmation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = (
    PROJECT_ROOT
    / "reports"
    / "optimization"
    / "postaudit_microfruit_v1"
    / "development"
    / "development_candidate_metrics.csv"
)
CONFIRMATION = (
    PROJECT_ROOT
    / "reports"
    / "optimization"
    / "postaudit_microfruit_v1"
    / "confirmation"
    / "POSTAUDIT_CONFIRMATION_REPORT.json"
)
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

INK = "#17212B"
MUTED = "#66727D"
GRID = "#D7DEE3"
BLUE = "#0072B2"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
LIGHT_BLUE = "#E8F3F8"
LIGHT_AMBER = "#FFF4DD"


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


def read_development() -> list[dict]:
    with DEVELOPMENT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "overall_AP50_95",
        "small_green_COCO_small_AP50_95",
        "small_green_COCO_small_AR50",
        "mean_latency_ms",
        "operating_precision",
        "mean_local_tiles",
    }
    return [
        {
            key: float(value) if key in numeric else value
            for key, value in row.items()
        }
        for row in rows
    ]


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)
    ax.grid(False)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=7.4,
        linespacing=1.18,
    )


def protocol_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    rounded_box(ax, (0.02, 0.59), 0.26, 0.27, "Cross-stage\nreference", LIGHT_BLUE, BLUE)
    rounded_box(
        ax,
        (0.37, 0.61),
        0.27,
        0.23,
        "Development\n6 candidates",
        "#F0F4F6",
        MUTED,
    )
    rounded_box(
        ax,
        (0.72, 0.61),
        0.26,
        0.23,
        "Selection lock\nF5 / MARS",
        LIGHT_AMBER,
        AMBER,
    )
    rounded_box(
        ax,
        (0.72, 0.12),
        0.26,
        0.25,
        "Held-back subset\none-time check",
        "#EAF6F1",
        GREEN,
    )
    for start, end in (
        ((0.28, 0.725), (0.37, 0.725)),
        ((0.64, 0.725), (0.72, 0.725)),
        ((0.85, 0.61), (0.85, 0.37)),
    ):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color=MUTED,
            )
        )
    ax.text(
        0.02,
        0.06,
        "Post hoc; known aggregate results → exploratory subset check.",
        transform=ax.transAxes,
        fontsize=6.2,
        color=MUTED,
        va="bottom",
    )
    ax.set_title("Frozen exploratory design", fontsize=9, color=INK, pad=6)
    panel_label(ax, "(a)", x=-0.04, y=1.07)


def development_panel(ax: plt.Axes, rows: list[dict]) -> None:
    colors = {"F0": MUTED, "F1": SKY, "F2": BLUE, "F3": VERMILLION, "F4": AMBER, "F5": GREEN}
    lookup = {row["candidate"]: row for row in rows}
    pareto = [lookup[name] for name in ("F0", "F1", "F5")]
    ax.plot(
        [row["mean_latency_ms"] for row in pareto],
        [row["small_green_COCO_small_AP50_95"] for row in pareto],
        color=MUTED,
        linewidth=0.9,
        linestyle=":",
        zorder=1,
        label="Latency–accuracy Pareto set",
    )
    for row in rows:
        name = row["candidate"]
        selected = name == "F5"
        ax.scatter(
            row["mean_latency_ms"],
            row["small_green_COCO_small_AP50_95"],
            s=74 if selected else 42,
            marker="*" if selected else "o",
            color=colors[name],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            name,
            (row["mean_latency_ms"], row["small_green_COCO_small_AP50_95"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.2,
            color=INK,
            fontweight="bold" if selected else "normal",
        )
    ax.axhline(
        next(row for row in rows if row["candidate"] == "F0")[
            "small_green_COCO_small_AP50_95"
        ],
        color=MUTED,
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xlabel("Mean development latency (ms/image)")
    ax.set_ylabel("Small-green COCO-small AP50-95")
    ax.set_title("Development accuracy-latency frontier", fontsize=9, color=INK, pad=6)
    ax.legend(frameon=False, loc="lower right", fontsize=6.6)
    style_axis(ax)
    panel_label(ax, "(b)")


def endpoint_panel(ax: plt.Axes, report: dict) -> None:
    f0, f5 = report["systems"]["F0"], report["systems"]["F5"]
    labels = ["Overall\nAP50-95", "Microfruit\nAP50-95", "Microfruit\nAR50"]
    f0_values = [
        f0["overall"]["AP50_95"],
        f0["small_green_COCO_small"]["AP50_95"],
        f0["small_green_COCO_small"]["AR50"],
    ]
    f5_values = [
        f5["overall"]["AP50_95"],
        f5["small_green_COCO_small"]["AP50_95"],
        f5["small_green_COCO_small"]["AR50"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, f0_values, width, color=MUTED, label="F0 balanced")
    ax.bar(x + width / 2, f5_values, width, color=GREEN, label="MARS recall")
    for position, value in zip(x - width / 2, f0_values, strict=True):
        ax.text(position, value + 0.018, f"{value:.3f}", ha="center", fontsize=6.8, color=INK)
    for position, value in zip(x + width / 2, f5_values, strict=True):
        ax.text(position, value + 0.018, f"{value:.3f}", ha="center", fontsize=6.8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Metric value")
    ax.set_title("One-time confirmation point estimates", fontsize=9, color=INK, pad=6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    style_axis(ax)
    panel_label(ax, "(c)")


def forest_panel(ax: plt.Axes, report: dict) -> None:
    keys = [
        ("overall_AP50_95", "Overall AP50-95"),
        ("all_COCO_small_AP50_95", "All small AP50-95"),
        ("small_green_COCO_small_AP50_95", "Microfruit AP50-95"),
        ("overall_AR50", "Overall AR50"),
        ("small_green_COCO_small_AR50", "Microfruit AR50"),
    ]
    rows = [report["paired_bootstrap_differences"][key] for key, _ in keys]
    labels = [label for _, label in keys]
    y = np.arange(len(rows))[::-1]
    differences = np.asarray([row["difference"] for row in rows])
    lower = np.asarray([row["lower_95"] for row in rows])
    upper = np.asarray([row["upper_95"] for row in rows])
    colors = [VERMILLION if low <= 0 <= high else GREEN for low, high in zip(lower, upper, strict=True)]
    for yi, estimate, low, high, color in zip(y, differences, lower, upper, colors, strict=True):
        ax.plot([low, high], [yi, yi], color=color, linewidth=1.8)
        ax.scatter(estimate, yi, s=33, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("MARS - F0 paired difference")
    ax.set_xlim(-0.10, 0.56)
    ax.set_title("95% image-cluster bootstrap intervals", fontsize=9, color=INK, pad=6)
    style_axis(ax, grid_axis="x")
    panel_label(ax, "(d)", x=-0.26)


def main() -> None:
    configure_style()
    rows = read_development()
    report = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.25, 5.25))
    grid = figure.add_gridspec(
        2,
        12,
        height_ratios=(0.92, 1.0),
        hspace=0.43,
        wspace=1.25,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.965,
    )
    protocol_panel(figure.add_subplot(grid[0, :5]))
    development_panel(figure.add_subplot(grid[0, 6:]), rows)
    endpoint_panel(figure.add_subplot(grid[1, :6]), report)
    forest_panel(figure.add_subplot(grid[1, 7:]), report)

    output_base = OUTPUT_DIR / "figS1_mars_postaudit_optimization_and_confirmation"
    figure.savefig(output_base.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"figure={output_base.resolve()}")


if __name__ == "__main__":
    main()
