#!/usr/bin/env python3
"""Compute the maintained paper figures and scientific SVG sections in a new root."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

# Support direct execution, including Python -I, from a source-only checkout.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import PIL
from PIL import Image
import sympy

import symergetics
from symergetics.computation.analysis import analyze_mathematical_patterns
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.visualization import get_config, set_config
from symergetics.visualization.advanced import (
    create_mega_graphical_abstract,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard,
)
from symergetics.visualization.geometry import (
    plot_polyhedron_3d,
    plot_quadray_coordinate,
)
from symergetics.visualization.mathematical import (
    plot_base_conversion,
    plot_continued_fraction_convergence,
    plot_pattern_analysis,
    plot_ssrcd_analysis,
)
from symergetics.visualization.numbers import plot_primorial_distribution


PATTERN_INPUTS = [
    121, 12321, 123454321, 12345678987654321,
    123, 456, 789, 12345678901234567890, 1001, 2002,
]


def json_value(value):
    """Retain exact rationals as strings and NumPy values as structured JSON."""
    if isinstance(value, (Fraction, sympy.Rational)):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported scientific metadata type: {type(value).__name__}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_record(path, output_dir):
    record = {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".svg":
        root = ET.parse(path).getroot()
        record.update(
            format="SVG", width=root.attrib["width"],
            height=root.attrib["height"], view_box=root.attrib["viewBox"],
        )
    else:
        with Image.open(path) as image:
            image.load()
            record.update(
                format=image.format, width_px=image.width, height_px=image.height,
                mode=image.mode, dpi=image.info.get("dpi"),
            )
    return record


def generate(output_dir):
    # Never reset or merge an existing output tree, even a partially generated one.
    output_dir.mkdir(parents=True, exist_ok=False)
    set_config({
        "output_dir": str(output_dir), "backend": "matplotlib",
        "organize_by_type": True, "include_timestamps": False,
        "figure_size": (10, 8), "dpi": 300, "formats": ["png"],
    })
    analyses = [
        analyze_mathematical_patterns(number, analysis_depth=4)
        for number in PATTERN_INPUTS
    ]
    records = []

    def produce(topic, producer, args, parameters, *, recorded_args=None,
                recorded_parameters=None, figure=None):
        result = producer(*args, **parameters)
        result["files"] = [
            Path(filename).resolve().relative_to(output_dir).as_posix()
            for filename in result["files"]
        ]
        record = {
            "topic": topic,
            "call": {
                "api": f"{producer.__module__}.{producer.__name__}",
                "args": args if recorded_args is None else recorded_args,
                "kwargs": parameters if recorded_parameters is None else recorded_parameters,
            },
            "result": result,
            "assets": [image_record(output_dir / name, output_dir)
                       for name in result["files"]],
        }
        if figure is not None:
            record["paper_figure"] = figure
        records.append(record)

    for figure, coordinates in ((1, (0, 0, 0, 0)), (2, (2, 1, 1, 0))):
        produce(
            "quadray_coordinate", plot_quadray_coordinate,
            [QuadrayCoordinate(*coordinates)],
            {"show_lattice": True, "lattice_size": 3, "backend": "matplotlib"},
            recorded_args=[{"type": "symergetics.core.coordinates.QuadrayCoordinate",
                            "args": coordinates}], figure=figure,
        )
    produce("continued_fraction_convergence", plot_continued_fraction_convergence,
            [3.14159], {"max_terms": 15, "backend": "matplotlib"}, figure=3)
    produce("base_conversion", plot_base_conversion, [30030],
            {"from_base": 10, "to_base": 2, "backend": "matplotlib"}, figure=4)
    produce("pattern_discovery", create_pattern_discovery_visualization, [analyses],
            {"title": "Selected Integer Pattern Analyses", "backend": "matplotlib"},
            figure=5)
    for figure, solid in ((6, "tetrahedron"), (7, "cube")):
        produce(solid, plot_polyhedron_3d, [solid], {"backend": "matplotlib"},
                figure=figure)
    produce("palindrome_coverage", plot_pattern_analysis, ["1001"],
            {"pattern_type": "palindrome", "backend": "matplotlib"}, figure=8)
    produce("ssrcd", plot_ssrcd_analysis, [3], {"backend": "matplotlib"}, figure=9)
    produce("primorial_distribution", plot_primorial_distribution, [],
            {"max_n": 13, "backend": "matplotlib"}, figure=10)
    produce("statistical_dashboard", create_statistical_analysis_dashboard, [analyses],
            {"title": "Selected Integer Statistical Analysis", "backend": "matplotlib"})
    for section in ("overview", "geometry", "arithmetic", "patterns", "all"):
        relative_path = f"scientific_sections/{section}.svg"
        produce(
            f"scientific_{section}", create_mega_graphical_abstract, [],
            {"output_path": str(output_dir / relative_path), "backend": "matplotlib",
             "section": section},
            recorded_parameters={"output_path": relative_path, "backend": "matplotlib",
                                 "section": section},
        )

    source_paths = sorted((SOURCE_ROOT / "symergetics").rglob("*.py"))
    source_paths.append(Path(__file__).resolve())
    config = get_config()
    config["output_dir"] = "."
    provenance = {
        "package": {"name": "symergetics", "version": symergetics.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rendering": {
            "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
            "config": config,
        },
        "toolchain": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "matplotlib_backend": matplotlib.get_backend(),
            "sympy": sympy.__version__, "pillow": PIL.__version__,
            "freetype": matplotlib.ft2font.__freetype_version__,
        },
        "sources": [{"path": path.relative_to(SOURCE_ROOT).as_posix(),
                     "sha256": sha256(path)} for path in source_paths],
        "analysis": {
            "api": "symergetics.computation.analysis.analyze_mathematical_patterns",
            "inputs": PATTERN_INPUTS, "analysis_depth": 4, "results": analyses,
        },
        "figures": records,
    }
    destination = output_dir / "provenance.json"
    destination.write_text(
        json.dumps(json_value(provenance), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(records)} scientific figures; provenance SHA256 {sha256(destination)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing paths are never overwritten")
    args = parser.parse_args()
    generate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
