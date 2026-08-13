"""Create Supplementary Figure S2 for frozen public CitDet validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    PROJECT_ROOT
    / "reports"
    / "external_validation"
    / "citdet_public_test_v1"
    / "CITDET_PUBLIC_EXTERNAL_VALIDATION_REPORT.json"
)
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

INK = "#17212B"
MUTED = "#7A8793"
GRID = "#D9E1E8"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
WHITE = "#FFFFFF"
SYSTEMS = ("D0", "G0", "MARS")
COLORS = {"D0": MUTED, "G0": BLUE, "MARS": VERMILLION}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


def absolute_metrics(ax: plt.Axes, report: dict) -> None:
    labels = ("AP50", "AP50–95", "AR50")
    x = np.arange(len(labels))
    width = 0.24
    for offset_index, name in enumerate(SYSTEMS):
        row = report["systems"][name]["COCO_small"]
        values = [row["AP50"], row["AP50_95"], row["AR50"]]
        positions = x + (offset_index - 1) * width
        bars = ax.bar(
            positions,
            values,
            width,
            color=COLORS[name],
            edgecolor=WHITE,
            linewidth=0.5,
            label=name,
            zorder=2,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Public-dataset COCO-small performance", fontsize=9, pad=7)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.02, 0.99))
    clean_axis(ax)
    panel_label(ax, "a")


def fixed_threshold(ax: plt.Axes, report: dict) -> None:
    labels = ("Precision", "Recall", "F1")
    x = np.arange(len(labels))
    width = 0.24
    for offset_index, name in enumerate(SYSTEMS):
        row = report["systems"][name]["operating_point_confidence_0_25_iou_0_50"]
        values = [row["precision"], row["recall"], row["F1"]]
        ax.bar(
            x + (offset_index - 1) * width,
            values,
            width,
            color=COLORS[name],
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=2,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Metric value")
    ax.set_title("Uncalibrated fixed operating point (conf. 0.25)", fontsize=9, pad=7)
    ax.text(
        0.98,
        0.97,
        "Count MAE: D0 22.37  |  G0 28.02  |  MARS 28.69",
        transform=ax.transAxes,
        fontsize=6.4,
        color=INK,
        ha="right",
        va="top",
    )
    clean_axis(ax)
    panel_label(ax, "b")


def forest_plot(ax: plt.Axes, report: dict) -> None:
    lookup = {
        (row["comparison"], row["size"], row["metric"]): row
        for row in report["paired_image_cluster_bootstrap"]["differences"]
    }
    entries = [
        ("G0 − D0", "Small AR50", lookup[("G0_minus_D0", "small", "AR50")], BLUE),
        ("G0 − D0", "Small AP50", lookup[("G0_minus_D0", "small", "AP50")], BLUE),
        ("G0 − D0", "Small AP50–95", lookup[("G0_minus_D0", "small", "AP50_95")], BLUE),
        ("MARS − G0", "Small AR50", lookup[("MARS_minus_G0", "small", "AR50")], VERMILLION),
        ("MARS − G0", "Small AP50", lookup[("MARS_minus_G0", "small", "AP50")], VERMILLION),
        ("MARS − G0", "Small AP50–95", lookup[("MARS_minus_G0", "small", "AP50_95")], VERMILLION),
    ]
    y = np.arange(len(entries))[::-1]
    for yi, (comparison, metric, row, color) in zip(y, entries, strict=True):
        low = float(row["lower_95"])
        high = float(row["upper_95"])
        estimate = float(row["difference"])
        interval_color = AMBER if low <= 0 <= high else color
        ax.plot([low, high], [yi, yi], color=interval_color, linewidth=1.8, zorder=2)
        ax.scatter(
            estimate,
            yi,
            s=33,
            color=interval_color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            -0.036,
            yi,
            f"{comparison}  |  {metric}",
            ha="right",
            va="center",
            fontsize=6.9,
        )
    ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_xlim(-0.075, 0.065)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Paired difference")
    ax.set_title("2,000-image-cluster bootstrap intervals", fontsize=9, pad=7)
    clean_axis(ax, "x")
    panel_label(ax, "c")


def efficiency_plane(ax: plt.Axes, report: dict) -> None:
    for name in SYSTEMS:
        system = report["systems"][name]
        latency = system["runtime"]["mean_latency_ms"]
        recall = system["COCO_small"]["AR50"]
        mae = system["operating_point_confidence_0_25_iou_0_50"]["count_MAE"]
        saturated = system["runtime"]["max_det_saturated_images"]
        ax.scatter(
            latency,
            recall,
            s=50 + mae * 2.2,
            color=COLORS[name],
            edgecolor=WHITE,
            linewidth=0.8,
            zorder=3,
        )
        # Keep every label clear of its bubble at manuscript scale.  The prior
        # MARS offset placed the marker over the leading digits of "102/119".
        offsets = {"D0": (5, -12), "G0": (-5, 18), "MARS": (-5, 18)}[name]
        horizontal_alignment = "right" if name in {"G0", "MARS"} else "left"
        ax.annotate(
            f"{name}\n{latency:.0f} ms; cap {saturated}/119",
            (latency, recall),
            xytext=offsets,
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color=COLORS[name],
            horizontalalignment=horizontal_alignment,
        )
    ax.plot(
        [report["systems"][name]["runtime"]["mean_latency_ms"] for name in SYSTEMS],
        [report["systems"][name]["COCO_small"]["AR50"] for name in SYSTEMS],
        color=GRID,
        linewidth=1.0,
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlim(25, 900)
    ax.set_xticks([30, 100, 300, 800], labels=["30", "100", "300", "800"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(0.78, 0.84)
    ax.set_xlabel("Mean end-to-end latency (ms/image, log scale)")
    ax.set_ylabel("COCO-small AR50")
    ax.set_title("Recall–latency trade-off", fontsize=9, pad=7)
    ax.text(
        0.03,
        0.05,
        "Bubble area scales with count MAE; cap denotes images reaching max_det=500",
        transform=ax.transAxes,
        fontsize=6.5,
        color=MUTED,
    )
    clean_axis(ax)
    panel_label(ax, "d")


def main() -> None:
    configure_style()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["status"] != "FROZEN_PUBLIC_EXTERNAL_VALIDATION_COMPLETE":
        raise RuntimeError("CitDet report is not complete")
    figure = plt.figure(figsize=(7.25, 6.45), facecolor=WHITE)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.98,
        bottom=0.09,
        top=0.95,
        wspace=0.38,
        hspace=0.42,
    )
    absolute_metrics(figure.add_subplot(grid[0, 0]), report)
    fixed_threshold(figure.add_subplot(grid[0, 1]), report)
    forest_plot(figure.add_subplot(grid[1, 0]), report)
    efficiency_plane(figure.add_subplot(grid[1, 1]), report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / "figS2_public_citdet_external_validation"
    figure.savefig(output_base.with_suffix(".png"), dpi=600, facecolor=WHITE)
    figure.savefig(output_base.with_suffix(".pdf"), facecolor=WHITE)
    figure.savefig(output_base.with_suffix(".svg"), facecolor=WHITE)
    plt.close(figure)
    print(f"output={output_base}")


if __name__ == "__main__":
    main()
