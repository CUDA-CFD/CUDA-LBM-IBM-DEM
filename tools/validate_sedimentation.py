#!/usr/bin/env python3
"""Validate CPU and CUDA sedimentation trajectories against the N=100 benchmark.

The utility uses only the Python standard library. It linearly interpolates the
CPU and CUDA particle-centre heights at the published reference times, reports
benchmark errors for both solvers, measures direct CPU-CUDA agreement, and
writes CSV, Markdown, and SVG outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Sample = tuple[float, float]
ReferenceComparison = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ErrorMetrics:
    count: int
    rmse: float
    relative_l2: float
    mean_abs_error: float
    max_abs_error: float


def read_reference(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "y_cm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Reference CSV must contain {sorted(required)}")
        for row in reader:
            rows.append((float(row["time_s"]), float(row["y_cm"])))
    return validate_samples(rows, path, minimum_rows=1)


def read_simulation(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "time_s" not in fields:
            raise ValueError(f"{path} must contain time_s")

        if "y_m" in fields:
            for row in reader:
                rows.append((float(row["time_s"]), 100.0 * float(row["y_m"])))
        elif "y_lu" in fields:
            # The benchmark resolution is 100 lattice cells per centimetre.
            for row in reader:
                rows.append((float(row["time_s"]), float(row["y_lu"]) / 100.0))
        else:
            raise ValueError(f"{path} must contain y_m or y_lu")

    return validate_samples(rows, path, minimum_rows=2)


def validate_samples(
    rows: list[Sample],
    path: Path,
    minimum_rows: int,
) -> list[Sample]:
    if len(rows) < minimum_rows:
        raise ValueError(f"{path} must contain at least {minimum_rows} data rows")

    rows.sort(key=lambda item: item[0])
    for index, (time, value) in enumerate(rows):
        if not math.isfinite(time) or not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value at data row {index + 1}")
        if index > 0 and time <= rows[index - 1][0]:
            raise ValueError(f"{path} time_s values must be strictly increasing")
    return rows


def interpolate(samples: Sequence[Sample], time: float) -> float:
    if time < samples[0][0] or time > samples[-1][0]:
        raise ValueError(
            f"Requested time {time:g} s lies outside simulation range "
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


def calculate_metrics(reference: Sequence[float], values: Sequence[float]) -> ErrorMetrics:
    if len(reference) != len(values) or not reference:
        raise ValueError("Metric inputs must be non-empty and have equal length")

    errors = [value - target for target, value in zip(reference, values)]
    squared_error = sum(error * error for error in errors)
    squared_reference = sum(target * target for target in reference)
    absolute_errors = [abs(error) for error in errors]

    relative_l2 = (
        math.sqrt(squared_error / squared_reference)
        if squared_reference > 0.0
        else (0.0 if squared_error == 0.0 else math.inf)
    )

    return ErrorMetrics(
        count=len(errors),
        rmse=math.sqrt(squared_error / len(errors)),
        relative_l2=relative_l2,
        mean_abs_error=sum(absolute_errors) / len(absolute_errors),
        max_abs_error=max(absolute_errors),
    )


def compare_at_reference_times(
    reference: Sequence[Sample],
    cpu: Sequence[Sample],
    gpu: Sequence[Sample],
) -> tuple[list[ReferenceComparison], ErrorMetrics, ErrorMetrics]:
    comparison: list[ReferenceComparison] = []
    reference_values: list[float] = []
    cpu_values: list[float] = []
    gpu_values: list[float] = []

    for time, reference_y in reference:
        cpu_y = interpolate(cpu, time)
        gpu_y = interpolate(gpu, time)
        comparison.append(
            (
                time,
                reference_y,
                cpu_y,
                cpu_y - reference_y,
                gpu_y,
                gpu_y - reference_y,
                gpu_y - cpu_y,
            )
        )
        reference_values.append(reference_y)
        cpu_values.append(cpu_y)
        gpu_values.append(gpu_y)

    return (
        comparison,
        calculate_metrics(reference_values, cpu_values),
        calculate_metrics(reference_values, gpu_values),
    )


def compare_cpu_gpu(
    cpu: Sequence[Sample],
    gpu: Sequence[Sample],
) -> tuple[list[Sample], ErrorMetrics]:
    overlap_start = max(cpu[0][0], gpu[0][0])
    overlap_end = min(cpu[-1][0], gpu[-1][0])
    if overlap_start > overlap_end:
        raise ValueError("CPU and CUDA trajectories have no overlapping time range")

    differences: list[Sample] = []
    cpu_values: list[float] = []
    gpu_values: list[float] = []
    for time, cpu_y in cpu:
        if overlap_start <= time <= overlap_end:
            gpu_y = interpolate(gpu, time)
            differences.append((time, gpu_y - cpu_y))
            cpu_values.append(cpu_y)
            gpu_values.append(gpu_y)

    if not differences:
        raise ValueError("CPU and CUDA trajectories have no comparable samples")

    return differences, calculate_metrics(cpu_values, gpu_values)


def write_comparison(path: Path, comparison: Iterable[ReferenceComparison]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "reference_y_cm",
                "cpu_y_cm",
                "cpu_error_cm",
                "gpu_y_cm",
                "gpu_error_cm",
                "gpu_minus_cpu_cm",
            ]
        )
        writer.writerows(comparison)


def write_report(
    path: Path,
    cpu_metrics: ErrorMetrics,
    gpu_metrics: ErrorMetrics,
    cpu_gpu_metrics: ErrorMetrics,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Sedimentation validation report",
                "",
                "## Benchmark comparison",
                "",
                "| Solver | Points | RMSE (cm) | Relative RMSE | Mean absolute error (cm) | Maximum absolute error (cm) |",
                "|---|---:|---:|---:|---:|---:|",
                format_metrics_row("CPU", cpu_metrics),
                format_metrics_row("CUDA", gpu_metrics),
                "",
                "## CPU-CUDA agreement",
                "",
                f"- Comparison samples: {cpu_gpu_metrics.count}",
                f"- RMSE: {cpu_gpu_metrics.rmse:.6g} cm",
                f"- Relative L2 difference: {100.0 * cpu_gpu_metrics.relative_l2:.6g}%",
                f"- Mean absolute difference: {cpu_gpu_metrics.mean_abs_error:.6g} cm",
                f"- Maximum absolute difference: {cpu_gpu_metrics.max_abs_error:.6g} cm",
                "",
                "The reference values are the N=100 particle-centre coordinates from "
                "Table III of Pu et al., Physics of Fluids 35, 053307 (2023).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def format_metrics_row(label: str, metrics: ErrorMetrics) -> str:
    return (
        f"| {label} | {metrics.count} | {metrics.rmse:.6g} | "
        f"{100.0 * metrics.relative_l2:.4f}% | "
        f"{metrics.mean_abs_error:.6g} | {metrics.max_abs_error:.6g} |"
    )


def downsample(points: Sequence[Sample], maximum_points: int = 1600) -> list[Sample]:
    if len(points) <= maximum_points:
        return list(points)
    stride = math.ceil((len(points) - 1) / (maximum_points - 1))
    sampled = list(points[::stride])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def axis_ticks(minimum: float, maximum: float, count: int = 5) -> list[float]:
    if maximum == minimum:
        return [minimum]
    return [minimum + index * (maximum - minimum) / (count - 1) for index in range(count)]


def svg_coordinates(
    points: Sequence[Sample],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> str:
    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def py(y: float) -> float:
        return height - bottom - (y - y_min) / (y_max - y_min) * (height - top - bottom)

    return " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)


def write_position_svg(
    path: Path,
    reference: Sequence[Sample],
    cpu: Sequence[Sample],
    gpu: Sequence[Sample],
) -> None:
    width, height = 860, 560
    left, right, top, bottom = 82, 35, 45, 72
    cpu_plot = downsample(cpu)
    gpu_plot = downsample(gpu)
    all_points = list(reference) + cpu_plot + gpu_plot
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_max == x_min:
        x_max = x_min + 1.0
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    ref_line = svg_coordinates(reference, x_min, x_max, y_min, y_max, width, height, left, right, top, bottom)
    cpu_line = svg_coordinates(cpu_plot, x_min, x_max, y_min, y_max, width, height, left, right, top, bottom)
    gpu_line = svg_coordinates(gpu_plot, x_min, x_max, y_min, y_max, width, height, left, right, top, bottom)

    x_ticks = axis_ticks(x_min, x_max)
    y_ticks = axis_ticks(y_min, y_max)

    elements = svg_frame(width, height, left, right, top, bottom, x_ticks, y_ticks, x_min, x_max, y_min, y_max, "Particle-centre height (cm)")
    elements.extend(
        [
            f'<polyline points="{cpu_line}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>',
            f'<polyline points="{gpu_line}" fill="none" stroke="#d62728" stroke-width="2" stroke-dasharray="8 5"/>',
            f'<polyline points="{ref_line}" fill="none" stroke="#111" stroke-width="2.6"/>',
        ]
    )
    for time, value in reference:
        x = left + (time - x_min) / (x_max - x_min) * (width - left - right)
        y = height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="#111"/>')

    elements.extend(
        [
            '<text x="110" y="70" font-family="sans-serif" font-size="14">Reference N=100</text>',
            '<line x1="72" y1="65" x2="102" y2="65" stroke="#111" stroke-width="2.6"/>',
            '<text x="110" y="94" font-family="sans-serif" font-size="14">CPU</text>',
            '<line x1="72" y1="89" x2="102" y2="89" stroke="#1f77b4" stroke-width="2.2"/>',
            '<text x="110" y="118" font-family="sans-serif" font-size="14">CUDA</text>',
            '<line x1="72" y1="113" x2="102" y2="113" stroke="#d62728" stroke-width="2" stroke-dasharray="8 5"/>',
        ]
    )
    write_svg_document(path, width, height, "Sedimentation trajectory validation", elements)


def write_difference_svg(path: Path, differences: Sequence[Sample]) -> None:
    width, height = 860, 480
    left, right, top, bottom = 96, 35, 45, 72
    points = downsample(differences)
    x_min = points[0][0]
    x_max = points[-1][0]
    if x_max == x_min:
        x_max = x_min + 1.0
    max_magnitude = max(abs(value) for _, value in points)
    y_limit = max(1.0e-16, 1.1 * max_magnitude)
    y_min, y_max = -y_limit, y_limit

    line = svg_coordinates(points, x_min, x_max, y_min, y_max, width, height, left, right, top, bottom)
    x_ticks = axis_ticks(x_min, x_max)
    y_ticks = axis_ticks(y_min, y_max)
    elements = svg_frame(width, height, left, right, top, bottom, x_ticks, y_ticks, x_min, x_max, y_min, y_max, "CUDA - CPU height (cm)", scientific_y=True)

    zero_y = height - bottom - (0.0 - y_min) / (y_max - y_min) * (height - top - bottom)
    elements.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#888" stroke-width="1"/>')
    elements.append(f'<polyline points="{line}" fill="none" stroke="#5b2c6f" stroke-width="2"/>')
    write_svg_document(path, width, height, "CPU-CUDA trajectory difference", elements)


def svg_frame(
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_label: str,
    scientific_y: bool = False,
) -> list[str]:
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    for value in x_ticks:
        x = left + (value - x_min) / (x_max - x_min) * (width - left - right)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#e5e5e5"/>',
                f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.2f}</text>',
            ]
        )
    for value in y_ticks:
        y = height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)
        label = f"{value:.2e}" if scientific_y else f"{value:.2f}"
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e5e5"/>',
                f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{(left + width - right)/2:.2f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Time (s)</text>',
            f'<text x="22" y="{height/2:.2f}" text-anchor="middle" transform="rotate(-90 22 {height/2:.2f})" font-family="sans-serif" font-size="14">{y_label}</text>',
        ]
    )
    return elements


def write_svg_document(
    path: Path,
    width: int,
    height: int,
    title: str,
    elements: Sequence[str],
) -> None:
    body = "\n  ".join(elements)
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{title}</title>
  {body}
</svg>
''',
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, default=Path("sedimentation.csv"))
    parser.add_argument("--gpu", type=Path, default=Path("sedimentation_gpu.csv"))
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
    try:
        reference = read_reference(args.reference)
        cpu = read_simulation(args.cpu)
        gpu = read_simulation(args.gpu)

        comparison, cpu_metrics, gpu_metrics = compare_at_reference_times(reference, cpu, gpu)
        cpu_gpu_differences, cpu_gpu_metrics = compare_cpu_gpu(cpu, gpu)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_comparison(args.output_dir / "validation_comparison.csv", comparison)
        write_report(
            args.output_dir / "validation_report.md",
            cpu_metrics,
            gpu_metrics,
            cpu_gpu_metrics,
        )
        write_position_svg(
            args.output_dir / "position_comparison.svg",
            reference,
            cpu,
            gpu,
        )
        write_difference_svg(
            args.output_dir / "cpu_gpu_difference.svg",
            cpu_gpu_differences,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Benchmark comparison")
    print(
        f"  CPU:  RMSE={cpu_metrics.rmse:.6g} cm, "
        f"relative RMSE={100.0 * cpu_metrics.relative_l2:.4f}%, "
        f"max error={cpu_metrics.max_abs_error:.6g} cm"
    )
    print(
        f"  CUDA: RMSE={gpu_metrics.rmse:.6g} cm, "
        f"relative RMSE={100.0 * gpu_metrics.relative_l2:.4f}%, "
        f"max error={gpu_metrics.max_abs_error:.6g} cm"
    )
    print("CPU-CUDA agreement")
    print(
        f"  RMSE={cpu_gpu_metrics.rmse:.6g} cm, "
        f"relative L2={100.0 * cpu_gpu_metrics.relative_l2:.6g}%, "
        f"max difference={cpu_gpu_metrics.max_abs_error:.6g} cm"
    )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
