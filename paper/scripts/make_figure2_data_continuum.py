"""Create Fig. 2: adverse plum-orchard scenes and spatial detection evidence.

Examples are fixed from manuscript/adverse_scene_ranking.csv using observable
scene criteria. Yellow reference boxes come from the completed human review;
cyan/red prediction overlays come from the frozen A5 checkpoint. The heatmaps
are Gaussian accumulations of detected centers weighted by confidence, not
feature-attribution or biological saliency maps.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "work" / "datasets" / "pruscope_human_240_v1" / "images" / "test"
ANNOTATION_ROOT = PROJECT_ROOT / "reports" / "human_annotation" / "pruscope_human_240_v1" / "annotations" / "internal_json"
PREDICTIONS = PROJECT_ROOT / "reports" / "evaluation" / "pruscope_a5_human_240_v1" / "frozen_predictions_remapped.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"

INK = "#17212B"
WHITE = "#FFFFFF"
YELLOW = "#F0E442"
CYAN = "#31C7E7"
RED = "#E69F00"


SCENES = (
    {"task": "T0169", "tag": "Backlight", "metric": "extreme 30%"},
    {"task": "T0081", "tag": "Deep shadow", "metric": "shadow 24%", "rotate_ccw": True},
    {"task": "T0111", "tag": "Defocus", "metric": "fruit 26"},
    {"task": "T0051", "tag": "Glare + occlusion", "metric": "fruit 6"},
    {"task": "T0029", "tag": "Leaf occlusion", "metric": "occluded 70%"},
    {"task": "T0032", "tag": "Dense mature", "metric": "fruit 224"},
    {"task": "T0189", "tag": "Dense microfruit", "metric": "fruit 84"},
    {"task": "T0047", "tag": "High dynamic range", "metric": "truncated 9"},
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_predictions() -> dict[str, dict]:
    result = {}
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        result[payload["audit_task_id"]] = payload
    return result


def load_reference(task: str) -> list[dict]:
    payload = json.loads((ANNOTATION_ROOT / f"{task}.json").read_text(encoding="utf-8"))
    return payload["annotations"]


def center_heatmap(height: int, width: int, predictions: list[dict]) -> np.ndarray:
    canvas_h = 320
    canvas_w = max(220, round(canvas_h * width / height))
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    heat = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for pred in predictions:
        confidence = float(pred["confidence"])
        if confidence < 0.05:
            continue
        x1, y1, x2, y2 = map(float, pred["xyxy"])
        cx = (x1 + x2) * 0.5 / width * canvas_w
        cy = (y1 + y2) * 0.5 / height * canvas_h
        box_ref = np.sqrt(max(1.0, (x2 - x1) * (y2 - y1))) / np.sqrt(width * height)
        sigma = float(np.clip(4.0 + 34.0 * box_ref * np.sqrt(canvas_w * canvas_h), 5.0, 18.0))
        heat += confidence * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def panel_label(ax: plt.Axes, text: str) -> None:
    label = ax.text(0.018, 0.982, text, transform=ax.transAxes, va="top", ha="left", color=WHITE, fontsize=8.2, fontweight="bold")
    label.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground=INK)])


def transform_xyxy_ccw(xyxy: list[float] | tuple[float, ...], original_width: int) -> list[float]:
    """Rotate an edge-coordinate box 90 degrees counterclockwise."""
    x1, y1, x2, y2 = map(float, xyxy)
    return [y1, original_width - x2, y2, original_width - x1]


def prepare_scene(scene: dict) -> tuple[np.ndarray, int, int, callable]:
    image = mpimg.imread(IMAGE_ROOT / f"{scene['task']}.jpg")
    original_height, original_width = image.shape[:2]
    if scene.get("rotate_ccw", False):
        image = np.rot90(image, 1)

        def transform(box):
            return transform_xyxy_ccw(box, original_width)

    else:

        def transform(box):
            return list(map(float, box))

    height, width = image.shape[:2]
    return image, height, width, transform


def add_metric_badge(ax: plt.Axes, text: str) -> None:
    """Use one quiet, consistent badge instead of multi-line caption boxes."""
    ax.text(
        0.975,
        0.025,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.25,
        fontweight="bold",
        color=WHITE,
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.08",
            "facecolor": INK,
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )


def add_reference_panel(ax: plt.Axes, scene: dict, panel: str) -> None:
    task = scene["task"]
    image, _, _, transform = prepare_scene(scene)
    ax.imshow(image)
    for annotation in load_reference(task):
        x1, y1, x2, y2 = transform(annotation["bbox_xyxy"])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=INK, linewidth=1.35))
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=YELLOW, linewidth=0.72))
    panel_label(ax, panel)
    add_metric_badge(ax, f"{scene['tag']} · {scene['metric']}")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color(INK)


def add_evidence_panel(ax: plt.Axes, scene: dict, predictions_by_task: dict[str, dict], panel: str) -> None:
    task = scene["task"]
    image, height, width, transform = prepare_scene(scene)
    predictions = []
    for source in predictions_by_task[task]["predictions"]:
        if float(source["confidence"]) < 0.05:
            continue
        transformed = dict(source)
        transformed["xyxy"] = transform(source["xyxy"])
        predictions.append(transformed)
    heat = center_heatmap(height, width, predictions)
    ax.imshow(image)
    ax.imshow(heat, cmap="turbo", alpha=np.where(heat > 0.08, np.minimum(0.68, heat * 0.72), 0.0), extent=(0, width, height, 0), interpolation="bilinear")
    shown = [p for p in predictions if float(p["confidence"]) >= 0.25]
    for pred in shown:
        x1, y1, x2, y2 = map(float, pred["xyxy"])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=CYAN, linewidth=0.8))
        ax.add_patch(Circle(((x1 + x2) / 2, (y1 + y2) / 2), radius=max(3.2, 0.0045 * min(width, height)), facecolor=RED, edgecolor=WHITE, linewidth=0.35))
    panel_label(ax, panel)
    add_metric_badge(ax, f"Detected {len(shown)} · conf ≥ .25")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color(INK)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    fig = plt.figure(figsize=(7.25, 10.25), facecolor="white")
    # Each panel keeps the source aspect ratio, so rotated landscape scenes
    # would otherwise sit at the top of an overly tall fixed grid cell.  A
    # height-ratio grid gives landscape rows a compact band and returns the
    # visual weight to the image evidence rather than empty whitespace.
    grid = fig.add_gridspec(4, 4, height_ratios=(1.0, 0.62, 1.0, 1.0), hspace=0.05, wspace=0.035)
    letters = "abcdefghijklmnop"
    for index, scene in enumerate(SCENES):
        row = index // 2
        pair = index % 2
        ref_col = pair * 2
        add_reference_panel(fig.add_subplot(grid[row, ref_col]), scene, f"({letters[index * 2]})")
        add_evidence_panel(fig.add_subplot(grid[row, ref_col + 1]), scene, predictions, f"({letters[index * 2 + 1]})")

    fig.text(0.5, 0.994, "Adverse plum-orchard scenes: reference–detection pairs", ha="center", va="top", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(0.5, 0.007, "Yellow = reference   Cyan = detection (conf ≥ .25)   Orange = center   Heat = confidence-weighted detection density", ha="center", va="bottom", fontsize=7.0, color=INK)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.975, bottom=0.028)

    stem = OUTPUT_DIR / "fig2_adverse_plum_scenes_and_detection_evidence"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"output={stem}")


if __name__ == "__main__":
    main()
