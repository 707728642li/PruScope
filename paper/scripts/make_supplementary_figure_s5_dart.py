"""Create Supplementary Figure S4 for PruScope-DART."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "optimization" / "dart_microfruit_v2" / "final" / "DART_FINAL_RESULTS.json"
OUTPUT = ROOT / "manuscript" / "figures" / "figS4_dart_microfruit_refinement.png"
OUTPUT_PDF = ROOT / "manuscript" / "figures" / "figS4_dart_microfruit_refinement.pdf"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    domains = ("internal", "plos", "citdet")
    labels = ("Internal", "External plum", "CitDet")
    colors = {"direct": "#0072B2", "score_only": "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)

    x = np.arange(len(domains))
    width = 0.34
    for axis, metric, title in (
        (axes[0, 0], "AP50_95", "Small-object AP50–95"),
        (axes[0, 1], "AR50", "Small-object AR50"),
    ):
        for offset, system in ((-width / 2, "direct"), (width / 2, "score_only")):
            values = [
                data["protected_domains"][domain]["systems"][system]["small"][metric]
                for domain in domains
            ]
            axis.bar(x + offset, values, width, label="A2 direct" if system == "direct" else "DART", color=colors[system])
        axis.set_xticks(x, labels)
        axis.set_ylabel(metric.replace("_", "–"))
        axis.set_title(title, loc="left", weight="bold")
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")

    bootstrap = {
        domain: {
            (row["size"], row["metric"]): row
            for row in data["protected_domains"][domain]["paired_direct_vs_score_only"]
        }
        for domain in domains
    }
    y = np.arange(len(domains))
    for index, metric in enumerate(("AP50_95", "AR50")):
        offset = -0.09 if index == 0 else 0.09
        rows = [bootstrap[domain][("small", metric)] for domain in domains]
        points = np.asarray([row["difference"] for row in rows])
        lower = np.asarray([row["lower_95"] for row in rows])
        upper = np.asarray([row["upper_95"] for row in rows])
        axes[1, 0].errorbar(
            points, y + offset,
            xerr=np.vstack([points - lower, upper - points]),
            fmt="o", capsize=3, label=metric.replace("_", "–"),
            color="#E69F00" if metric == "AP50_95" else "#0072B2",
        )
    axes[1, 0].axvline(0, color="#303030", lw=1)
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].set_xlabel("DART − direct paired difference (95% bootstrap CI)")
    axes[1, 0].set_title("Paired source-image uncertainty", loc="left", weight="bold")
    axes[1, 0].grid(False)
    axes[1, 0].spines[["top", "right"]].set_visible(False)
    axes[1, 0].legend(frameon=False, loc="lower right")

    dense_rows = []
    for domain in domains:
        dense = next(
            row for row in data["protected_domains"][domain]["density_strata"]
            if row["stratum"] == "dense"
        )
        dense_rows.append(dense)
    dense_ap = [row["systems"]["dart"]["small"]["AP50_95"] - row["systems"]["direct"]["small"]["AP50_95"] for row in dense_rows]
    dense_ar = [row["systems"]["dart"]["small"]["AR50"] - row["systems"]["direct"]["small"]["AR50"] for row in dense_rows]
    axes[1, 1].bar(x - width / 2, dense_ap, width, label="Δ AP50–95", color="#E69F00")
    axes[1, 1].bar(x + width / 2, dense_ar, width, label="Δ AR50", color="#0072B2")
    axes[1, 1].axhline(0, color="#303030", lw=1)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("DART − direct")
    axes[1, 1].set_title("Ground-truth-density upper tertile", loc="left", weight="bold")
    axes[1, 1].grid(False)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    axes[1, 1].legend(frameon=False, ncol=2, loc="upper left")

    for label, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.text(-0.13, 1.08, f"({label})", transform=axis.transAxes, fontsize=13, weight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())
    print(OUTPUT_PDF.resolve())


if __name__ == "__main__":
    main()
