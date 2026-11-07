#!/usr/bin/env python3
"""Exercise the actual QuadMath, symergetics and QuadCraft coordinate APIs.

Run in symergetics' scientific environment with sibling checkouts, Node.js and
an available C++17 compiler. Paths and executable names can be overridden with
--help. Native adapter files are compiled in a temporary directory, not a repo.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Sequence


NODE_ADAPTER = r"""
const fs = require('node:fs');
const {Quadray} = require(process.argv[1]);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(([values, otherValues]) => {
    const q = new Quadray(...values), other = new Quadray(...otherValues);
    const alias = new Quadray(...values.map(v => v + 19));
    const p = q.toCartesian(), n = q.normalized();
    const back = Quadray.fromCartesian(p.x, p.y, p.z).normalized();
    return [p.x, p.y, p.z, q.length(), Quadray.distance(q, alias),
        Quadray.distance(q, other), n.a, n.b, n.c, n.d,
        back.a, back.b, back.c, back.d];
})));
"""

NATIVE_ADAPTER = r"""
#include "Quadray.h"
#include <iomanip>
#include <iostream>
int main() {
    using QuadCraft::Quadray;
    float a, b, c, d, e, f, g, h;
    std::cout << std::setprecision(9);
    while (std::cin >> a >> b >> c >> d >> e >> f >> g >> h) {
        const Quadray q(a,b,c,d), other(e,f,g,h), alias(a+19,b+19,c+19,d+19);
        const auto p = q.toCartesian();
        const auto n = q.normalized();
        const auto back = Quadray::fromCartesian(p).normalized();
        std::cout << p.x << ' ' << p.y << ' ' << p.z << ' ' << q.length()
            << ' ' << q.distance(alias) << ' ' << q.distance(other)
            << ' ' << n.a << ' ' << n.b << ' ' << n.c << ' ' << n.d
            << ' ' << back.a << ' ' << back.b << ' ' << back.c << ' ' << back.d << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
"""


def run(command: list[str], stdin: str = "") -> str:
    """Run a real runtime; retain its diagnostics on failure."""
    result = subprocess.run(command, input=stdin, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}:\n{result.stderr or result.stdout}")
    return result.stdout


def suite_xyz(values: Sequence[int]) -> tuple[int, int, int]:
    """The suite's scale-1 XYZ convention, used as an independent oracle."""
    a, b, c, d = values
    return a - b - c + d, a + b - c - d, a - b + c - d


def check_row(label: str, row: Sequence[float], values: Sequence[int],
              other: Sequence[int], craft: bool = False) -> None:
    """Check units, gauge invariance, a distinct-point distance and inversion."""
    xyz = suite_xyz(values)
    other_xyz = suite_xyz(other)
    scale = 1 / math.sqrt(2) if craft else 1.0
    if craft:
        xyz = xyz[0], xyz[2], xyz[1]
        other_xyz = other_xyz[0], other_xyz[2], other_xyz[1]
    minimum = min(values)
    canonical = [v - minimum for v in values]
    expected = [*(v * scale for v in xyz), math.hypot(*xyz) * scale, 0.0,
                math.dist(xyz, other_xyz) * scale, *canonical, *canonical]
    if len(row) != len(expected):
        raise AssertionError(f"{label}: expected {len(expected)} result fields, got {len(row)}")
    for index, (actual, wanted) in enumerate(zip(row, expected)):
        if not math.isclose(actual, wanted, rel_tol=2e-5, abs_tol=2e-5):
            raise AssertionError(f"{label}: q={values}, field={index}, expected {wanted}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--quadmath-root", type=Path, default=here.parent / "QuadMath")
    parser.add_argument("--quadcraft-root", type=Path, default=here.parent / "QuadCraft")
    parser.add_argument("--node", default="node", help="Node.js executable")
    parser.add_argument("--cxx", default="c++", help="C++17 compiler executable")
    args = parser.parse_args()
    math_root = args.quadmath_root.resolve()
    craft_root = args.quadcraft_root.resolve()
    for root, required in ((math_root, "src/quadmath/core/quadray.py"),
                           (craft_root, "games/4d_generic/quadray.js")):
        if not (root / required).is_file():
            parser.error(f"missing checkout file: {root / required}")
    sys.path[:0] = [str(here), str(math_root / "src"), str(craft_root)]

    import numpy as np

    from quadmath.core.quadray import (
        DEFAULT_EMBEDDING, Quadray, ace_tetravolume_5x5, centroid_exact, distance,
        integer_tetra_volume, magnitude, quadray_from_xyz, to_xyz,
    )
    from quadmath.lattice.conversions import xyz_to_quadray_canonical
    from quadmath.lattice.omni_numbering import cumulative_count, site_at_index
    from symergetics.core.coordinates import QuadrayCoordinate, enumerate_ivm_sites
    from symergetics.core.numbers import SymergeticsNumber
    from symergetics.geometry.polyhedra import (
        SymergeticsPolyhedron, ace_tetra_volume_5x5, integer_tetra_volume as symer_volume,
    )
    from games.src.space.quadrays import Quadray as CraftQuadray

    # Independently implemented FCC graph balls must exchange the same physical
    # sites even though each project retains its own numbering convention.
    ivm_sites = set()
    ivm_site_comparisons = 0
    for radius in range(4):
        expected = {site_at_index(index, max_shell=radius).as_tuple()
                    for index in range(cumulative_count(radius))}
        actual_sites = {q.as_tuple() for q in enumerate_ivm_sites(radius)}
        if actual_sites != expected:
            raise AssertionError(f"IVM graph ball mismatch at radius {radius}")
        ivm_sites.update(actual_sites)
        ivm_site_comparisons += len(actual_sites)

    # Exercise both the full integer image and FCC sites beyond the small box,
    # with a signed gauge offset on every coordinate representative.
    sites = sorted({tuple(v - min(q) for v in q) for q in product(range(4), repeat=4)}
                   | ivm_sites)
    cases = [(tuple(v + i % 7 - 3 for v in q), sites[(i + 37) % len(sites)])
             for i, q in enumerate(sites)]
    for values, other in cases:
        alias = tuple(v + 19 for v in values)
        q, p = Quadray(*values), Quadray(*other)
        xyz = to_xyz(q, DEFAULT_EMBEDDING)
        back = quadray_from_xyz(*xyz, DEFAULT_EMBEDDING)
        check_row("QuadMath", [*xyz, magnitude(q, DEFAULT_EMBEDDING),
                  distance(q, Quadray(*alias), DEFAULT_EMBEDDING),
                  distance(q, p, DEFAULT_EMBEDDING),
                  *q.normalize().as_tuple(), *back.as_tuple()], values, other)
        if xyz_to_quadray_canonical(xyz) != q.normalize():
            raise AssertionError(f"QuadMath exact inverse failed for {values}")

        q, p = QuadrayCoordinate(*values), QuadrayCoordinate(*other)
        xyz = q.to_xyz()
        check_row("symergetics", [*xyz, q.magnitude(),
                  q.distance_to(QuadrayCoordinate(*alias)), q.distance_to(p),
                  *q.as_tuple(), *QuadrayCoordinate.from_xyz(*xyz).as_tuple()], values, other)

        q, p = CraftQuadray(*values), CraftQuadray(*other)
        xyz = q.to_xyz()
        check_row("QuadCraft Python", [*xyz, q.length(),
                  q.distance_to(CraftQuadray(*alias)), q.distance_to(p),
                  *q.normalized().as_tuple(),
                  *CraftQuadray.from_xyz(*xyz).normalized().as_tuple()], values, other, craft=True)

    node_rows = json.loads(run([args.node, "-e", NODE_ADAPTER,
                               str(craft_root / "games/4d_generic/quadray.js")], json.dumps(cases)))
    with tempfile.TemporaryDirectory(prefix="synergetics-suite-") as temporary:
        directory = Path(temporary)
        source, binary = directory / "adapter.cpp", directory / "adapter"
        source.write_text(NATIVE_ADAPTER)
        run([args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
             "-I", str(craft_root / "src/core/coordinate"), str(source), "-o", str(binary)])
        stdin = "".join(" ".join(map(str, (*q, *p))) + "\n" for q, p in cases)
        native_rows = [list(map(float, line.split())) for line in run([str(binary)], stdin).splitlines()]
    for label, rows in (("QuadCraft JavaScript", node_rows), ("QuadCraft C++", native_rows)):
        if len(rows) != len(cases):
            raise AssertionError(f"{label}: expected {len(cases)} rows, got {len(rows)}")
        for row, (values, other) in zip(rows, cases):
            check_row(label, row, values, other, craft=True)

    # Both determinant formulations must agree, including beyond binary64's
    # integer precision. Vertex-wise gauge shifts must not change the volume.
    rng = random.Random(0)
    tetrahedra = [((0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)),
                  ((0,0,0,0), (2,1,0,1), (2,1,1,0), (2,0,1,1))]
    tetrahedra += [tuple(tuple(rng.randrange(-8, 9) for _ in range(4)) for _ in range(4))
                   for _ in range(32)]
    for i, vertices in enumerate(tetrahedra):
        expected = integer_tetra_volume(*(Quadray(*q) for q in vertices))
        if i < 2 and expected != (Fraction(1, 4), Fraction(1))[i]:
            raise AssertionError("reference tetrahedron units disagree")
        shifted = [tuple(v + 10**20 + j for v in q) for j, q in enumerate(vertices)]
        actual = [ace_tetravolume_5x5(*(Quadray(*q) for q in shifted)),
                  symer_volume(*(QuadrayCoordinate(*q) for q in shifted)),
                  ace_tetra_volume_5x5(*(QuadrayCoordinate(*q) for q in shifted))]
        if any(volume != expected for volume in actual):
            raise AssertionError(f"exact volume case {i}: expected {expected}, got {actual}")

    # Fractional centers are exchanged as rationals, never through an integer
    # constructor or a binary64 transport. The oracle averages raw representatives;
    # implementations may canonicalize each input first, which must be equivalent.
    centroid_groups = [
        *tetrahedra,
        ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)),
        ((10**400 + 1, 0, 0, 0), (0, 0, 0, 0)),
        ((-10**450, -10**450, -10**450, -10**450),),
    ]
    for index, vertices in enumerate(centroid_groups):
        mean = tuple(Fraction(sum(q[axis] for q in vertices), len(vertices))
                     for axis in range(4))
        minimum = min(mean)
        expected = tuple(value - minimum for value in mean)
        shifted = [tuple(value + (-1)**j * (10**500 + j) for value in q)
                   for j, q in enumerate(vertices)]
        for representatives in (vertices, shifted):
            actual = [
                centroid_exact(Quadray(*q) for q in representatives),
                SymergeticsPolyhedron(
                    [QuadrayCoordinate(*q) for q in representatives],
                ).centroid_exact(),
            ]
            if any(center != expected for center in actual):
                raise AssertionError(
                    f"exact centroid case {index}: expected {expected}, got {actual}",
                )

    # Fraction can retain NumPy integer fields. Both libraries must promote
    # these before accumulation, snapping or powers, not after overflow.
    rational_ingress = [np.int64(2**62), np.uint64(2**63)]
    for value in rational_ingress:
        rational = Fraction(value, 1)
        math_point = Quadray(rational, 0, 0, 0)
        symer_point = QuadrayCoordinate(rational, 0, 0, 0)
        expected = (Fraction(int(value)), Fraction(0), Fraction(0), Fraction(0))
        centers = (
            centroid_exact([math_point, math_point]),
            SymergeticsPolyhedron([symer_point, symer_point]).centroid_exact(),
        )
        power = (SymergeticsNumber(value) ** Fraction(np.int64(2), 1)).value
        if any(center != expected for center in centers) or power != int(value)**2:
            raise AssertionError(f"fixed-width rational ingress failed for {value}")
    print(json.dumps({"status": "passed", "coordinate_cases_per_runtime": len(cases),
                      "runtimes": ["QuadMath", "symergetics", "QuadCraft Python",
                                   "QuadCraft JavaScript", "QuadCraft C++"],
                      "exact_volume_cases": len(tetrahedra),
                      "ivm_site_comparisons": ivm_site_comparisons,
                      "exact_centroid_cases": len(centroid_groups),
                      "native_rational_ingress_cases": len(rational_ingress),
                      "exact_centroid_libraries": ["QuadMath", "symergetics"]}, indent=2))


if __name__ == "__main__":
    main()
