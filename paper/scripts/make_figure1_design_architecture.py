"""Synchronize publication Figure 1 from the authoritative workflow artwork.

The high-resolution ``model_workflow.png`` is the reviewed scientific artwork.
This script preserves that raster exactly for the PNG export and creates
matching PDF and SVG wrappers without redrawing or reinterpreting the model.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY_ROOT / "paper" / "figure_sources" / "model_workflow.png"
OUTPUT_DIR = REPOSITORY_ROOT / "paper" / "figures"
STEM = "fig1_experimental_design_and_pruscope_architecture"


def write_pdf(source: Path, output: Path, width_px: int, height_px: int, dpi: float) -> None:
    width_pt = width_px / dpi * 72.0
    height_pt = height_px / dpi * 72.0
    pdf = canvas.Canvas(str(output), pagesize=(width_pt, height_pt), pageCompression=1)
    pdf.drawImage(
        ImageReader(str(source)),
        0,
        0,
        width=width_pt,
        height=height_pt,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.showPage()
    pdf.save()


def write_svg(source: Path, output: Path, width_px: int, height_px: int) -> None:
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">\n'
        "  <title>Experimental safeguards and unified scale-stage PruScope architecture</title>\n"
        f'  <image width="{width_px}" height="{height_px}" xlink:href="data:image/png;base64,{encoded}"/>\n'
        "</svg>\n"
    )
    output.write_text(svg, encoding="utf-8", newline="\n")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Authoritative Figure 1 source is missing: {SOURCE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{STEM}.png"
    pdf = OUTPUT_DIR / f"{STEM}.pdf"
    svg = OUTPUT_DIR / f"{STEM}.svg"

    shutil.copyfile(SOURCE, png)
    with Image.open(SOURCE) as image:
        width_px, height_px = image.size
        dpi_info = image.info.get("dpi", (600.0, 600.0))
    dpi = float(dpi_info[0]) if dpi_info and dpi_info[0] else 600.0
    write_pdf(SOURCE, pdf, width_px, height_px, dpi)
    write_svg(SOURCE, svg, width_px, height_px)

    print(f"Figure 1 synchronized from {SOURCE}")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"SVG: {svg}")


if __name__ == "__main__":
    main()
