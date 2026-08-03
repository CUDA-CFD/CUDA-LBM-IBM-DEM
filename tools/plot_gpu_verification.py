#!/usr/bin/env python3
"""Plot CPU-GPU verification errors using only the Python standard library."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    source = Path("results/gpu_verification.csv")
    destination = Path("results/gpu_verification.svg")

    if not source.exists():
        raise SystemExit(f"Missing verification file: {source}")

    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("Verification CSV contains no data rows.")

    steps = [float(row["step"]) for row in rows]
    series = {
        "rho": [max(float(row["rho_max"]), 1.0e-30) for row in rows],
        "ux": [max(float(row["ux_max"]), 1.0e-30) for row in rows],
        "uy": [max(float(row["uy_max"]), 1.0e-30) for row in rows],
        "f": [max(float(row["f_max"]), 1.0e-30) for row in rows],
    }

    import math

    width, height = 900, 520
    left, right, top, bottom = 90, 30, 35, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    xmin, xmax = min(steps), max(steps)
    logs = [math.log10(value) for values in series.values() for value in values]
    ymin, ymax = min(logs), max(logs)
    if ymax == ymin:
        ymax = ymin + 1.0

    def sx(value: float) -> float:
        return left + (value - xmin) / max(xmax - xmin, 1.0) * plot_width

    def sy(value: float) -> float:
        return top + (ymax - math.log10(value)) / (ymax - ymin) * plot_height

    palette = {"rho": "#1f77b4", "ux": "#ff7f0e", "uy": "#2ca02c", "f": "#d62728"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">Timestep</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="sans-serif">Maximum absolute error</text>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">CPU–GPU LBM verification</text>',
    ]

    for name, values in series.items():
        points = " ".join(f"{sx(step):.2f},{sy(value):.2f}" for step, value in zip(steps, values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{palette[name]}" stroke-width="2"/>')

    legend_x = left + plot_width - 120
    for index, name in enumerate(series):
        y = top + 20 + index * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{palette[name]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+32}" y="{y+5}" font-family="sans-serif">{name}</text>')

    lines.append("</svg>")
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
