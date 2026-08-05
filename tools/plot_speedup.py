#!/usr/bin/env python3
"""Plot CPU and CUDA runtime results from the complete pipeline benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/performance/speedup_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/performance/speedup_comparison.svg"),
    )
    args = parser.parse_args()

    try:
        rows = read_rows(args.input)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    runtimes: dict[str, list[float]] = {"CPU": [], "CUDA": []}
    for row in rows:
        backend = row["backend"]
        if backend in runtimes:
            runtimes[backend].append(float(row["runtime_seconds"]))

    if not runtimes["CPU"] or not runtimes["CUDA"]:
        print("Benchmark CSV must contain CPU and CUDA rows.", file=sys.stderr)
        return 1

    cpu = statistics.median(runtimes["CPU"])
    gpu = statistics.median(runtimes["CUDA"])
    speedup = cpu / gpu

    args.output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 760, 480
    left, right, top, bottom = 110, 40, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(cpu, gpu) * 1.12

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value / maximum)

    bars = [("CPU", cpu, left + plot_width * 0.25),
            ("CUDA", gpu, left + plot_width * 0.68)]
    bar_width = 145

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:700}.axis{font-size:13px}.label{font-size:16px;font-weight:600}.value{font-size:14px}</style>',
        f'<text x="{width/2}" y="34" text-anchor="middle" class="title">Complete pipeline CPU–CUDA performance</text>',
        f'<text x="{width/2}" y="58" text-anchor="middle" class="axis">Median runtime; speedup = {speedup:.2f}×</text>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#444"/>',
    ]

    for tick in range(6):
        value = maximum * tick / 5
        yy = y(value)
        lines.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-12}" y="{yy+5:.2f}" text-anchor="end" class="axis">{value:.2f}</text>')

    for name, value, center in bars:
        yy = y(value)
        hh = top + plot_height - yy
        lines.append(f'<rect x="{center-bar_width/2:.2f}" y="{yy:.2f}" width="{bar_width}" height="{hh:.2f}" rx="5"/>')
        lines.append(f'<text x="{center:.2f}" y="{top+plot_height+30}" text-anchor="middle" class="label">{name}</text>')
        lines.append(f'<text x="{center:.2f}" y="{yy-10:.2f}" text-anchor="middle" class="value">{value:.6f} s</text>')

    lines.append(f'<text transform="translate(24 {top+plot_height/2}) rotate(-90)" text-anchor="middle" class="axis">Runtime (seconds)</text>')
    lines.append('</svg>')

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Plot: {args.output}")
    print(f"Median speedup: {speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
