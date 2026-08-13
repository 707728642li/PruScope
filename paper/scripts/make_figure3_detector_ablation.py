"""Create Fig. 3: detector ablation, paired effects, and efficiency."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "manuscript" / "final_metrics_registry.json"
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

INK = "#17212B"
MUTED = "#566573"
BLUE = "#0072B2"
SKY = "#56B4E9"
TEAL = "#009E73"
AMBER = "#E69F00"
PURPLE = "#7A5195"
LIGHT_GRAY = "#D8DEE3"
PALE_GRAY = "#F4F6F7"

ARCHITECTURES = ("a0", "a1", "a2", "a5")
ARCH_LABELS = ("A0\nNative", "A1\n+P2", "A2\n+CCPH", "A5\n+OSSA")
ARCH_COLORS = (MUTED, SKY, TEAL, BLUE)


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
    ax.grid(False)


def size_value(registry: dict, domain: str, model: str, size: str) -> float:
    corrected = registry.get("corrected_multiseed") or {}
    for row in corrected.get("architecture_summary", []):
        if (
            row["domain"] == domain
            and row["architecture"] == model
            and row["size"] == size
        ):
            return float(row["mean"])
    return primary_seed_value(registry, domain, model, size)


def primary_seed_value(registry: dict, domain: str, model: str, size: str) -> float:
    corrected = registry.get("corrected_multiseed") or {}
    for row in corrected.get("seed_level_metrics", []):
        if (
            row["domain"] == domain
            and row["architecture"] == model
            and int(row["seed"]) == 20260805
            and row["size"] == size
        ):
            return float(row["AP50_95"])
    for row in registry["size_stratified"][domain]:
        if row["model"] == model and row["size"] == size:
            return float(row["AP50_95"])
    raise KeyError((domain, model, size))


def corrected_seed_values(
    registry: dict, domain: str, architecture: str, size: str
) -> list[float]:
    payload = registry.get("corrected_multiseed")
    if not payload:
        return []
    values = []
    for row in payload["seed_level_metrics"]:
        if (
            row["domain"] != domain
            or row["size"] != size
            or row["architecture"] != architecture
        ):
            continue
        values.append(float(row["AP50_95"]))
    return values


def plot_ablation(ax: plt.Axes, registry: dict, size: str, title: str) -> None:
    x = np.arange(len(ARCHITECTURES))
    width = 0.34
    for offset, domain, label, hatch in (
        (-width / 2, "internal", "Internal", None),
        (width / 2, "external_plum", "External plum", "///"),
    ):
        values = [size_value(registry, domain, model, size) for model in ARCHITECTURES]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=ARCH_COLORS,
            edgecolor="white" if hatch is None else INK,
            linewidth=0.5,
            hatch=hatch,
            alpha=0.92,
            zorder=2,
            label=label,
        )
        for architecture in ARCHITECTURES:
            model_index = ARCHITECTURES.index(architecture)
            seed_values = corrected_seed_values(registry, domain, architecture, size)
            if seed_values:
                jitter = np.linspace(-0.035, 0.035, len(seed_values))
                ax.scatter(
                    model_index + offset + jitter,
                    seed_values,
                    s=15,
                    facecolors="white",
                    edgecolors=INK,
                    linewidths=0.55,
                    zorder=4,
                )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.018 if size == "small" else 0.012),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
                rotation=0,
                zorder=5,
            )
    ax.set_xticks(x, ARCH_LABELS)
    ax.set_ylabel("AP50–95")
    lower = 0.28 if size == "small" else 0.48
    upper = 0.48 if size == "small" else 0.64
    ax.set_ylim(lower, upper)
    ax.set_title(title, fontsize=9, color=INK, pad=24 if size == "small" else 7)
    style_axis(ax)
    if size == "small":
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=7,
            handlelength=1.5,
            columnspacing=1.2,
        )


def bootstrap_effect(
    registry: dict, key: str, size: str, candidate: str
) -> tuple[float, float, float]:
    report = registry["paired_bootstrap"][key]
    if not report:
        raise RuntimeError(f"Missing paired bootstrap: {key}")
    for row in report["paired_differences"]:
        if row["candidate"] == candidate and row["size"] == size and row["metric"] == "AP50_95":
            return (
                100.0 * float(row["difference"]),
                100.0 * float(row["lower_95"]),
                100.0 * float(row["upper_95"]),
            )
    raise KeyError((key, size))


def seed_mean_difference(
    registry: dict, contrast: str, domain: str, size: str
) -> float | None:
    payload = registry.get("corrected_multiseed")
    if not payload:
        return None
    for row in payload["paired_difference_summary"]:
        if (
            row.get("contrast", "a5_minus_a0") == contrast
            and row["domain"] == domain
            and row["size"] == size
        ):
            return 100.0 * float(row["mean"])
    return None


def plot_effects(ax: plt.Axes, registry: dict) -> None:
    entries = (
        ("Internal – all", "internal", "all"),
        ("Internal – small", "internal", "small"),
        ("External – all", "external_plum", "all"),
        ("External – small", "external_plum", "small"),
    )
    contrasts = (
        {
            "name": "a2_minus_a1",
            "label": "A2 − A1 (capacity preservation)",
            "candidate": "a2",
            "internal_key": "corrected_a2_vs_a1_internal",
            "external_key": "corrected_a2_vs_a1_external_plum",
            "color": TEAL,
            "marker": "o",
            "offset": 0.13,
        },
        {
            "name": "a5_minus_a2",
            "label": "A5 − A2 (OSSA increment)",
            "candidate": "a5",
            "internal_key": "a5_vs_a2_internal",
            "external_key": "a5_vs_a2_external_plum",
            "color": PURPLE,
            "marker": "s",
            "offset": -0.13,
        },
    )
    y = np.arange(len(entries))[::-1]
    ax.axvline(0, color=INK, linewidth=0.8, linestyle="--", zorder=1)
    for row_y, (_, domain, size) in zip(y, entries, strict=True):
        for contrast in contrasts:
            key = contrast["internal_key"] if domain == "internal" else contrast["external_key"]
            estimate, lower, upper = bootstrap_effect(
                registry, key, size, contrast["candidate"]
            )
            point_y = row_y + contrast["offset"]
            ax.errorbar(
                estimate,
                point_y,
                xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                fmt=contrast["marker"],
                color=contrast["color"],
                ecolor=contrast["color"],
                capsize=2.2,
                markersize=4.5,
                linewidth=1.1,
                zorder=3,
            )
            seed_mean = seed_mean_difference(
                registry, contrast["name"], domain, size
            )
            if seed_mean is not None:
                ax.scatter(
                    seed_mean,
                    point_y,
                    marker=contrast["marker"],
                    s=36,
                    facecolors="white",
                    edgecolors=contrast["color"],
                    linewidths=0.9,
                    zorder=4,
                )
    ax.set_yticks(y, [entry[0] for entry in entries])
    ax.set_xlabel("AP50–95 difference (percentage points)")
    ax.set_title("Capacity preservation and OSSA increment (95% CI)", fontsize=9, color=INK, pad=7)
    for contrast in contrasts:
        ax.scatter(
            [],
            [],
            marker=contrast["marker"],
            color=contrast["color"],
            label=contrast["label"],
        )
    ax.legend(
        frameon=False,
        fontsize=6.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def plot_efficiency(ax: plt.Axes, registry: dict) -> None:
    runtime = registry.get("runtime")
    if not runtime:
        raise RuntimeError("Runtime benchmark is not available")
    runtime_lookup = {row["model"]: row for row in runtime["results"]}
    label_offsets = {
        "a0": (-9, -15, "right"),
        "a1": (5, 4, "left"),
        "a2": (7, 8, "left"),
        "a5": (5, 4, "left"),
    }
    for model, label, color in zip(ARCHITECTURES, ARCH_LABELS, ARCH_COLORS, strict=True):
        row = runtime_lookup[model]
        latency = float(row["latency_ms_per_image_mean"])
        latency_sd = float(row["latency_ms_per_image_sd"])
        accuracy = primary_seed_value(registry, "external_plum", model, "small")
        parameters_m = float(row["parameters"]) / 1e6
        ax.errorbar(
            latency,
            accuracy,
            xerr=latency_sd,
            fmt="none",
            ecolor=color,
            capsize=2,
            linewidth=0.9,
            zorder=2,
        )
        ax.scatter(
            latency,
            accuracy,
            s=35 + 2.0 * parameters_m,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy, horizontal_alignment = label_offsets[model]
        ax.annotate(
            label.split("\n")[0],
            (latency, accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            color=INK,
            ha=horizontal_alignment,
        )
    ax.set_xlabel("Latency (ms/image; batch 1)")
    ax.set_ylabel("External small-object AP50–95")
    ax.set_title("Direct-inference accuracy–cost plane", fontsize=9, color=INK, pad=7)
    style_axis(ax)


def main() -> None:
    configure_style()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if any(value is None for value in registry["paired_bootstrap"].values()):
        raise RuntimeError("Paired bootstrap registry is incomplete")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)
    plot_ablation(axes[0, 0], registry, "all", "Overall direct-inference accuracy")
    plot_ablation(axes[0, 1], registry, "small", "Small-object direct-inference accuracy")
    plot_effects(axes[1, 0], registry)
    plot_efficiency(axes[1, 1], registry)
    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)"), strict=True):
        add_panel_label(ax, label)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "fig3_detector_ablation_statistics_and_efficiency"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(stem.resolve())


if __name__ == "__main__":
    main()
