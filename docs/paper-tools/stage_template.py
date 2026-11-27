"""Stage the canonical manuscript and check its declared mathematical examples."""

import json
import re
from fractions import Fraction
from pathlib import Path
from shutil import copyfile

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber
from symergetics.computation.primorials import primorial, scheherazade_power
from symergetics.geometry.polyhedra import Cube, Cuboctahedron, Octahedron, Tetrahedron

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/manuscript"
OUTPUT = ROOT / "output"
FIGURE = re.compile(r"!\[Figure (\d+): ([^\n]+)\]\((\.\./\.\./output/publication/1\.4\.0/[^)]+)\)")
SECTION = re.compile(r"^#{1,2} (.+)$", re.MULTILINE)
LINK = re.compile(r"(?<!!)\]\(([^ )\n]+\.md)(?:#([^ )\n]+))?\)")


def arithmetic():
    rational = SymergeticsNumber(Fraction(3, 4)) + SymergeticsNumber(Fraction(1, 6))
    assert rational.value == Fraction(11, 12)
    point = QuadrayCoordinate(2, 1, 1, 0)
    assert point.as_tuple() == QuadrayCoordinate(3, 2, 2, 1).as_tuple()
    assert list(point.to_xyz()) == [0, 2, 2]
    volumes = [item().volume() for item in (Tetrahedron, Cube, Octahedron, Cuboctahedron)]
    assert volumes == [1, 3, 4, 20]
    products = [primorial(bound).value for bound in (6, 10, 13)]
    assert products == [30, 210, 30030]
    powers = [scheherazade_power(exponent).value for exponent in (1, 2, 3)]
    assert powers == [1001, 1002001, 1003003001]
    return {
        "rational": {"numerator": rational.value.numerator, "denominator": rational.value.denominator},
        "coordinates": list(point.as_tuple()),
        "volumes_ivm": [int(value) for value in volumes],
        "prime_product_bounds": [6, 10, 13],
        "prime_products": [int(value) for value in products],
        "scheherazade_powers": [int(value) for value in powers],
        "base": 1001,
        "block_radix": 10**3,
    }


def main():
    manuscript = OUTPUT / "manuscript"
    figures = OUTPUT / "figures"
    if manuscript.exists() or figures.exists():
        raise SystemExit("Use a clean checkout without output/manuscript or output/figures.")
    results = arithmetic()
    manuscript.mkdir(parents=True)
    figures.mkdir()
    sections = sorted(SOURCE.glob("[0-9][0-9]_*.md"))
    anchors = {section.name: f"sec:{section.stem[3:].replace('_', '-')}" for section in sections}
    registry = {}
    for section in sections:
        text = section.read_text(encoding="utf-8")
        if section.name != "99_references.md":
            text = SECTION.sub(lambda item: f"# {item[1]} {{#{anchors[section.name]}}}", text, count=1)
        text = LINK.sub(
            lambda item: (
                f"](index.html#{item[2] or anchors[item[1]]})"
                if item[1] in anchors
                else f"](https://github.com/docxology/symergetics/blob/main/"
                f"{(SOURCE / item[1]).resolve().relative_to(ROOT).as_posix()}"
                f"{'#' + item[2] if item[2] else ''})"
            ),
            text,
        )
        for number, caption, address in FIGURE.findall(text):
            original = (SOURCE / address).resolve()
            label = f"fig:symergetics-{number}"
            copyfile(original, figures / original.name)
            registry[label] = {
                "label": label,
                "filename": original.name,
                "generated_by": original.relative_to(ROOT).as_posix(),
                "caption": caption,
                "alt_text": caption,
            }
            text = text.replace(
                f"![Figure {number}: {caption}]({address})",
                f"![{caption}](../figures/{original.name}){{#{label}}}",
            )
        (manuscript / section.name).write_text(text, encoding="utf-8")
    assert len(registry) == 10
    (figures / "figure_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    (manuscript / "config.yaml").write_text(
        "# Generated manuscript ordering\n" + (SOURCE / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    copyfile(SOURCE / "references.bib", manuscript / "references.bib")
    (OUTPUT / "mathematical-checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Verified rational arithmetic, common-gauge coordinates, four IVM volumes, primorials, and powers of 1001.")


if __name__ == "__main__":
    main()
