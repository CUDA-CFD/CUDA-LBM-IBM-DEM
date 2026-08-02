#!/usr/bin/env python3
"""Compare CudaLBM-IBM-DEM sedimentation output with Table III (N=100).

The script uses only the Python standard library. It linearly interpolates the
simulated particle-centre height at the reference times, writes a comparison
CSV, a Markdown report, and a compact SVG plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def read_reference(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "y_cm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Reference CSV must contain {sorted(required)}")
        for row in reader:
            rows.append((float(row["time_s"]), float(row["y_cm"])))
    if not rows:
        raise ValueError("Reference CSV is empty")
    return rows


def read_simulation(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "time_s" not in fields:
            raise ValueError("Simulation CSV must contain time_s")
        if "y_m" in fields:
            for row in reader:
                rows.append((float(row["time_s"]), 100.0 * float(row["y_m"])))
        elif "y_lu" in fields:
            # Benchmark resolution is 100 lattice cells per centimetre.
            for row in reader:
                rows.append((float(row["time_s"]), float(row["y_lu"]) / 100.0))
        else:
            raise ValueError("Simulation CSV must contain y_m or y_lu")
    if len(rows) < 2:
        raise ValueError("Simulation CSV must contain at least two rows")
    rows.sort(key=lambda item: item[0])
    return rows


def interpolate(samples: list[tuple[float, float]], time: float) -> float:
    if time < samples[0][0] or time > samples[-1][0]:
        raise ValueError(
            f"Reference time {time:g} s lies outside simulation range "
            f"[{samples[0][0]:g}, {samples[-1][0]:g}] s"
        )

    lo = 0
    hi = len(samples) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if samples[mid][0] < time:
            lo = mid + 1
        else:
            hi = mid - 1

    upper = min(lo, len(samples) - 1)
    if samples[upper][0] == time or upper == 0:
        return samples[upper][1]

    lower = upper - 1
    t0, y0 = samples[lower]
    t1, y1 = samples[upper]
    fraction = (time - t0) / (t1 - t0)
    return y0 + fraction * (y1 - y0)


def write_comparison(
    path: Path,
    comparison: Iterable[tuple[float, float, float, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "reference_y_cm", "simulation_y_cm", "error_cm"])
        writer.writerows(comparison)


def write_report(
    path: Path,
    count: int,
    rmse: float,
    relative_rmse: float,
    max_abs_error: float,
    mean_abs_error: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Sedimentation validation report",
                "",
                f"- Comparison points: {count}",
                f"- RMSE: {rmse:.6g} cm",
                f"- Relative RMSE: {100.0 * relative_rmse:.4f}%",
                f"- Mean absolute error: {mean_abs_error:.6g} cm",
                f"- Maximum absolute error: {max_abs_error:.6g} cm",
                "",
                "The reference values are the N=100 particle-centre coordinates from "
                "Table III of Pu et al., Physics of Fluids 35, 053307 (2023).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def svg_polyline(
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margin: int,
) -> str:
    def px(x: float) -> float:
        return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)

    def py(y: float) -> float:
        return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)

    return " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)


def write_svg(
    path: Path,
    reference: list[tuple[float, float]],
    simulation: list[tuple[float, float]],
) -> None:
    width, height, margin = 760, 500, 65
    all_points = reference + simulation
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    ref_line = svg_polyline(reference, x_min, x_max, y_min, y_max, width, height, margin)
    sim_line = svg_polyline(simulation, x_min, x_max, y_min, y_max, width, height, margin)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
  <polyline points="{ref_line}" fill="none" stroke="#222" stroke-width="3"/>
  <polyline points="{sim_line}" fill="none" stroke="#777" stroke-width="2" stroke-dasharray="8 5"/>
  <text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif">Time (s)</text>
  <text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif">Particle-centre height (cm)</text>
  <text x="{margin+15}" y="{margin+18}" font-family="sans-serif">Reference N=100</text>
  <line x1="{margin+160}" y1="{margin+13}" x2="{margin+205}" y2="{margin+13}" stroke="#222" stroke-width="3"/>
  <text x="{margin+15}" y="{margin+43}" font-family="sans-serif">CudaLBM-IBM-DEM</text>
  <line x1="{margin+160}" y1="{margin+38}" x2="{margin+205}" y2="{margin+38}" stroke="#777" stroke-width="2" stroke-dasharray="8 5"/>
  <text x="{margin}" y="{height-margin+22}" font-family="sans-serif" font-size="12">{x_min:g}</text>
  <text x="{width-margin}" y="{height-margin+22}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max:g}</text>
  <text x="{margin-8}" y="{height-margin}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.2f}</text>
  <text x="{margin-8}" y="{margin+4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.2f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, default=Path("sedimentation.csv"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("benchmark/sedimentation/reference_position_N100.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sedimentation_validation")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = read_reference(args.reference)
    simulation = read_simulation(args.simulation)

    comparison: list[tuple[float, float, float, float]] = []
    squared_error = 0.0
    squared_reference = 0.0
    absolute_errors: list[float] = []

    for time, reference_y in reference:
        simulation_y = interpolate(simulation, time)
        error = simulation_y - reference_y
        comparison.append((time, reference_y, simulation_y, error))
        squared_error += error * error
        squared_reference += reference_y * reference_y
        absolute_errors.append(abs(error))

    rmse = math.sqrt(squared_error / len(comparison))
    relative_rmse = math.sqrt(squared_error / squared_reference)
    max_abs_error = max(absolute_errors)
    mean_abs_error = sum(absolute_errors) / len(absolute_errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison(args.output_dir / "validation_comparison.csv", comparison)
    write_report(
        args.output_dir / "validation_report.md",
        len(comparison),
        rmse,
        relative_rmse,
        max_abs_error,
        mean_abs_error,
    )
    write_svg(
        args.output_dir / "position_comparison.svg",
        reference,
        [(row[0], row[2]) for row in comparison],
    )

    print(f"RMSE: {rmse:.6g} cm")
    print(f"Relative RMSE: {100.0 * relative_rmse:.4f}%")
    print(f"Mean absolute error: {mean_abs_error:.6g} cm")
    print(f"Maximum absolute error: {max_abs_error:.6g} cm")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
