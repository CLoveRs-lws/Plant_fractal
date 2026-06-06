#!/usr/bin/env python3
"""Generate plant-like branching structures with an L-system.

The script uses only the Python standard library and writes an SVG image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PRESETS = {
    "classic": {
        "axiom": "X",
        "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        "angle": 25.0,
        "iterations": 5,
    },
    "bushy": {
        "axiom": "F",
        "rules": {"F": "F[+F]F[-F][F]"},
        "angle": 22.5,
        "iterations": 4,
    },
    "binary": {
        "axiom": "X",
        "rules": {"X": "F[-X][+X]", "F": "FF"},
        "angle": 28.0,
        "iterations": 7,
    },
}

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def rewrite(axiom: str, rules: Dict[str, str], iterations: int) -> str:
    """Apply all production rules in parallel for a fixed number of steps."""
    state = axiom
    for _ in range(iterations):
        state = "".join(rules.get(symbol, symbol) for symbol in state)
    return state


def interpret(
    commands: Iterable[str],
    angle_degrees: float,
    step: float = 1.0,
) -> List[Segment]:
    """Interpret an L-system string as turtle graphics."""
    x, y = 0.0, 0.0
    heading = 90.0
    stack: List[Tuple[float, float, float]] = []
    segments: List[Segment] = []

    for command in commands:
        if command == "F":
            radians = math.radians(heading)
            next_x = x + step * math.cos(radians)
            next_y = y + step * math.sin(radians)
            segments.append(((x, y), (next_x, next_y)))
            x, y = next_x, next_y
        elif command == "+":
            heading += angle_degrees
        elif command == "-":
            heading -= angle_degrees
        elif command == "[":
            stack.append((x, y, heading))
        elif command == "]":
            if not stack:
                raise ValueError("Unmatched closing bracket in L-system string")
            x, y, heading = stack.pop()

    if stack:
        raise ValueError("Unmatched opening bracket in L-system string")
    return segments


def render_svg(
    segments: Sequence[Segment],
    output_path: Path,
    width: int = 1000,
    height: int = 1000,
    margin: int = 50,
) -> None:
    """Scale line segments to a canvas and save them as an SVG image."""
    if not segments:
        raise ValueError("The L-system produced no drawable segments")

    points = [point for segment in segments for point in segment]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def transform(point: Point) -> Point:
        px = margin + (point[0] - min_x) * scale
        py = height - margin - (point[1] - min_y) * scale
        return px, py

    path_commands = []
    for start, end in segments:
        x1, y1 = transform(start)
        x2, y2 = transform(end)
        path_commands.append(f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}")

    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                '<path d="'
                + " ".join(path_commands)
                + '" fill="none" stroke="#245c32" stroke-width="1.8" '
                + 'stroke-linecap="round" stroke-linejoin="round"/>'
            ),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a plant-like SVG using a deterministic L-system."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="classic",
        help="L-system rule set to use (default: classic)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the preset iteration count",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Override the preset turning angle in degrees",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("l_system_plant.svg"),
        help="Output SVG path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = PRESETS[args.preset]
    iterations = preset["iterations"] if args.iterations is None else args.iterations
    angle = preset["angle"] if args.angle is None else args.angle

    if not 1 <= iterations <= 9:
        raise ValueError("iterations must be between 1 and 9")

    commands = rewrite(preset["axiom"], preset["rules"], iterations)
    segments = interpret(commands, angle)
    render_svg(segments, args.output)

    print(f"Preset: {args.preset}")
    print(f"Iterations: {iterations}")
    print(f"Angle: {angle:g} degrees")
    print(f"Rewritten symbols: {len(commands)}")
    print(f"Drawn segments: {len(segments)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
