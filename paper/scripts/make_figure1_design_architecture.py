"""Create Plant Phenomics Fig. 1 as editable vector and 600-dpi bitmap.

The figure intentionally distinguishes cross-sectional stage coverage from
longitudinal tracking, separates the primary A2/A5/GLAF lineages, and shows the
validation-frozen post-review DART route without implying fresh confirmation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

INK = "#17212B"
MUTED = "#566573"
LINE = "#87929C"
BLUE = "#0072B2"
SKY = "#56B4E9"
TEAL = "#009E73"
AMBER = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#7A5195"
LIGHT_BLUE = "#EAF4FA"
LIGHT_TEAL = "#E8F5F0"
LIGHT_AMBER = "#FFF3D6"
LIGHT_PURPLE = "#F2ECF7"
LIGHT_GRAY = "#F3F5F6"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.4,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str = WHITE,
    edge: str = LINE,
    text_color: str = INK,
    fontsize: float = 8.0,
    weight: str = "normal",
    radius: float = 0.018,
    linewidth: float = 0.9,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    connectionstyle: str = "arc3",
    linewidth: float = 1.0,
    mutation_scale: float = 8.5,
    linestyle: str = "solid",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=1,
        )
    )


def panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.005, 0.985, "(a)", va="top", ha="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.075,
        0.982,
        "Experimental and technical design",
        va="top",
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    # Cohort strip. Circle size conveys apparent fruit scale without implying
    # repeated observation of the same fruit.
    cohort_y = 0.795
    centers = (0.18, 0.50, 0.82)
    radii = (0.028, 0.045, 0.060)
    fills = (SKY, TEAL, AMBER)
    labels = (
        "Small-green plum\ncanopy microfruit",
        "Medium-green plum\nexpanding fruit",
        "Mature plum\ncolor-developed fruit",
    )
    for x, radius, fill, label in zip(centers, radii, fills, labels, strict=True):
        ax.add_patch(Circle((x, cohort_y), radius, facecolor=fill, edgecolor=INK, linewidth=0.8))
        ax.text(x, cohort_y - 0.085, label, ha="center", va="top", fontsize=7.5, color=INK)
    ax.text(
        0.50,
        0.895,
        "Cross-sectional stage coverage; no same-fruit longitudinal series",
        ha="center",
        va="center",
        fontsize=7.8,
        color=VERMILLION,
        fontstyle="italic",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5, "alpha": 0.96},
        zorder=4,
    )

    rounded_box(
        ax,
        (0.07, 0.425),
        0.34,
        0.095,
        "One-class plum localization\nshared detector across stages",
        face=LIGHT_BLUE,
        edge=BLUE,
        fontsize=8.0,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.59, 0.425),
        0.34,
        0.095,
        "Ordered stage phenotyping\nROI crops + cohort stage labels",
        face=LIGHT_PURPLE,
        edge=PURPLE,
        fontsize=8.0,
        weight="bold",
    )
    arrow(ax, (0.50, 0.565), (0.24, 0.525), connectionstyle="arc3,rad=0.16")
    arrow(ax, (0.50, 0.565), (0.76, 0.525), connectionstyle="arc3,rad=-0.16")

    rounded_box(
        ax,
        (0.06, 0.245),
        0.88,
        0.095,
        "Capture-group-disjoint train / validation / test\nProtected stage-test images excluded from detector ancestry",
        face=LIGHT_GRAY,
        edge=INK,
        fontsize=8.2,
        weight="bold",
    )
    arrow(ax, (0.24, 0.42), (0.38, 0.345))
    arrow(ax, (0.76, 0.42), (0.62, 0.345))

    rounded_box(
        ax,
        (0.06, 0.055),
        0.38,
        0.105,
        "Independent reference-box tests\nInternal high-resolution + PLOS plum",
        face=LIGHT_TEAL,
        edge=TEAL,
        fontsize=8.0,
    )
    rounded_box(
        ax,
        (0.56, 0.055),
        0.38,
        0.105,
        "Validation-only selection\nPrimary thresholds + post-review DART grid",
        face=LIGHT_AMBER,
        edge=AMBER,
        fontsize=8.0,
    )
    arrow(ax, (0.38, 0.24), (0.25, 0.165))
    arrow(ax, (0.62, 0.24), (0.75, 0.165))


def panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.005, 0.985, "(b)", va="top", ha="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.075,
        0.982,
        "Primary PruScope architecture and evaluation lineages",
        va="top",
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    ax.text(
        0.73,
        0.930,
        "A2 = P2 + CCPH   |   A5 = A2 + OSSA",
        ha="center",
        va="center",
        fontsize=7.1,
        color=MUTED,
        fontstyle="italic",
    )
    rounded_box(ax, (0.02, 0.745), 0.11, 0.095, "RGB orchard\nimage", face=LIGHT_GRAY, edge=INK)
    rounded_box(
        ax,
        (0.17, 0.69),
        0.18,
        0.20,
        "YOLO26m backbone\n\nP2  stride 4\nP3  stride 8\nP4  stride 16\nP5  stride 32",
        face=LIGHT_BLUE,
        edge=BLUE,
        fontsize=7.6,
        weight="bold",
    )
    arrow(ax, (0.13, 0.792), (0.165, 0.792))

    rounded_box(
        ax,
        (0.40, 0.805),
        0.15,
        0.090,
        "OSSA\nA5-only attention",
        face=LIGHT_TEAL,
        edge=TEAL,
        fontsize=7.5,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.60, 0.735),
        0.17,
        0.125,
        "CCPH\ncapacity-preserving\nP2-P5 detection head",
        face=LIGHT_PURPLE,
        edge=PURPLE,
        fontsize=7.7,
        weight="bold",
    )
    # The lower route is A2 (backbone -> CCPH). OSSA is an explicit optional
    # upper branch used only by A5, so the topology cannot be misread as A2
    # passing through OSSA.
    arrow(ax, (0.35, 0.755), (0.595, 0.775), connectionstyle="arc3,rad=0.04")
    ax.text(0.475, 0.735, "A2 direct path", ha="center", va="center", fontsize=6.6, color=BLUE)
    arrow(ax, (0.35, 0.835), (0.395, 0.850))
    arrow(ax, (0.55, 0.850), (0.595, 0.825))
    rounded_box(ax, (0.82, 0.745), 0.16, 0.095, "Direct fruit boxes\n+ confidence", face=WHITE, edge=INK)
    arrow(ax, (0.77, 0.792), (0.815, 0.792))

    # Global-local branch and validation-frozen scene gate.
    rounded_box(
        ax,
        (0.13, 0.485),
        0.20,
        0.105,
        "Global PruScope-Det\ncontext + large fruit",
        face=LIGHT_BLUE,
        edge=BLUE,
    )
    rounded_box(
        ax,
        (0.13, 0.315),
        0.20,
        0.105,
        "Local PruScope-Det tiles\nmicrofruit recovery",
        face=LIGHT_AMBER,
        edge=AMBER,
    )
    arrow(ax, (0.075, 0.74), (0.125, 0.54), connectionstyle="arc3,rad=0.20")
    arrow(ax, (0.075, 0.74), (0.125, 0.37), connectionstyle="arc3,rad=0.12")
    rounded_box(
        ax,
        (0.40, 0.397),
        0.18,
        0.14,
        "GLAF\nfrozen A5 scene gate +\narea-aware fusion",
        face=LIGHT_TEAL,
        edge=TEAL,
        weight="bold",
    )
    arrow(ax, (0.33, 0.54), (0.395, 0.495))
    arrow(ax, (0.33, 0.37), (0.395, 0.44))

    rounded_box(
        ax,
        (0.66, 0.41),
        0.17,
        0.115,
        "Unified fruit\nlocalization",
        face=WHITE,
        edge=INK,
        weight="bold",
    )
    arrow(ax, (0.58, 0.467), (0.655, 0.467))

    rounded_box(
        ax,
        (0.64, 0.17),
        0.22,
        0.125,
        "DCOH\nordered ROI head\n2 cumulative logits",
        face=LIGHT_PURPLE,
        edge=PURPLE,
        weight="bold",
    )
    arrow(ax, (0.745, 0.405), (0.745, 0.30))

    rounded_box(
        ax,
        (0.04, 0.055),
        0.46,
        0.12,
        "Phenotypes: count  |  apparent-size distribution\nsmall / medium / large AP  |  confidence",
        face=LIGHT_GRAY,
        edge=INK,
        fontsize=7.8,
    )
    rounded_box(
        ax,
        (0.61, 0.005),
        0.30,
        0.105,
        "Stage probabilities\n+ developmental index [0, 2]",
        face=LIGHT_AMBER,
        edge=AMBER,
        fontsize=7.8,
    )
    arrow(ax, (0.66, 0.425), (0.47, 0.18), connectionstyle="arc3,rad=0.16")
    arrow(ax, (0.75, 0.165), (0.76, 0.115))

    # Small visual cue that stage probabilities are ordered.
    triangle = Polygon(
        [(0.93, 0.15), (0.98, 0.15), (0.98, 0.29)],
        closed=True,
        facecolor=AMBER,
        edgecolor=INK,
        linewidth=0.7,
        alpha=0.85,
    )
    ax.add_patch(triangle)
    ax.text(0.956, 0.30, "ordered", ha="center", va="bottom", fontsize=6.8, color=MUTED)


def panel_c(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.005, 0.985, "(c)", va="top", ha="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.075,
        0.982,
        "Optional post-review PruScope-DART high-recall route",
        va="top",
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.50,
        0.895,
        "Validation-frozen; all three protected domains were already known (descriptive evidence)",
        ha="center",
        va="center",
        fontsize=7.1,
        color=VERMILLION,
        fontstyle="italic",
    )

    rounded_box(
        ax,
        (0.015, 0.53),
        0.18,
        0.23,
        "Difficult-target A2\nP2 + CCPH (no OSSA)\n\n1 × 1024 global\n+ gated 768 tiles",
        face=LIGHT_BLUE,
        edge=BLUE,
        fontsize=7.2,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.245, 0.555),
        0.17,
        0.18,
        "Cross-view proposals\ncluster at IoU 0.55\n\nsplit by global\nobservations",
        face=LIGHT_GRAY,
        edge=INK,
        fontsize=7.1,
        weight="bold",
    )
    arrow(ax, (0.195, 0.645), (0.24, 0.645))

    rounded_box(
        ax,
        (0.465, 0.68),
        0.18,
        0.105,
        "Global anchors\nretain boxes + scores exactly",
        face=LIGHT_BLUE,
        edge=BLUE,
        fontsize=7.1,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.465, 0.465),
        0.18,
        0.145,
        "Local-only candidates\narea ≤ 4096\n224 crop + 16 metadata",
        face=LIGHT_AMBER,
        edge=AMBER,
        fontsize=7.0,
        weight="bold",
    )
    arrow(ax, (0.415, 0.66), (0.46, 0.725), connectionstyle="arc3,rad=-0.12")
    arrow(ax, (0.415, 0.62), (0.46, 0.545), connectionstyle="arc3,rad=0.12")

    rounded_box(
        ax,
        (0.70, 0.45),
        0.18,
        0.17,
        "DART tail\nResNet-18 + metadata MLP\nobjectness + uncertainty\n(score weight 0.25)",
        face=LIGHT_PURPLE,
        edge=PURPLE,
        fontsize=7.0,
        weight="bold",
    )
    arrow(ax, (0.645, 0.535), (0.695, 0.535))

    rounded_box(
        ax,
        (0.74, 0.68),
        0.22,
        0.105,
        "Anchor-preserving merge\nadd local-only boxes; NMS 0.55",
        face=LIGHT_TEAL,
        edge=TEAL,
        fontsize=7.0,
        weight="bold",
    )
    arrow(ax, (0.645, 0.73), (0.735, 0.73))
    arrow(ax, (0.79, 0.625), (0.84, 0.675), connectionstyle="arc3,rad=-0.10")
    ax.text(
        0.685,
        0.395,
        "Box offsets and uncertainty penalty were disabled in the frozen main mode",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
        fontstyle="italic",
    )

    balanced = rounded_box(
        ax,
        (0.10, 0.095),
        0.32,
        0.15,
        "Balanced real-time mode\nA2 direct  |  49.6 images/s",
        face=LIGHT_GRAY,
        edge=INK,
        fontsize=7.5,
        weight="bold",
    )
    balanced.set_linestyle("--")
    high_recall = rounded_box(
        ax,
        (0.56, 0.075),
        0.36,
        0.19,
        "High-recall offline mode\nA2 + DART  |  1.16 images/s\nsmall-object AR50 increased in all 3 known domains",
        face=LIGHT_TEAL,
        edge=TEAL,
        fontsize=7.2,
        weight="bold",
    )
    high_recall.set_linestyle("--")
    arrow(ax, (0.10, 0.53), (0.26, 0.25), connectionstyle="arc3,rad=0.18", linestyle="--")
    arrow(ax, (0.85, 0.675), (0.75, 0.27), connectionstyle="arc3,rad=-0.12", color=TEAL)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 8.40), facecolor=WHITE)
    grid = fig.add_gridspec(3, 1, height_ratios=(0.72, 0.96, 0.84), hspace=0.055)
    panel_a(fig.add_subplot(grid[0]))
    panel_b(fig.add_subplot(grid[1]))
    panel_c(fig.add_subplot(grid[2]))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.985, bottom=0.025)

    stem = OUTPUT_DIR / "fig1_experimental_design_and_pruscope_architecture"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
