"""Canonical FCC shells and honest transformation visualization failures."""

import ast
import base64
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from symergetics.core.coordinates import QuadrayCoordinate, enumerate_ivm_sites
from symergetics.visualization import get_config, set_config
from symergetics.visualization.geometry import (
    plot_coordinate_transformation,
    plot_ivm_lattice,
    plot_quadray_coordinate,
)


def _fcc_xyz(radius):
    """Independent Cartesian FCC cuboctahedral graph-ball oracle.

    In half-XYZ units FCC sites have even coordinate sum; graph distance is
    max(L-infinity, L-one/2), because a step changes two coordinates by one.
    """
    return {
        (2 * x, 2 * y, 2 * z)
        for x, y, z in product(range(-radius, radius + 1), repeat=3)
        if (x + y + z) % 2 == 0 and abs(x) + abs(y) + abs(z) <= 2 * radius
    }


def _shell(site):
    xyz = site.to_xyz()
    return int(max(max(abs(v) for v in xyz) / 2, sum(abs(v) for v in xyz) / 4))


@pytest.fixture
def output_config(tmp_path):
    original = get_config()
    set_config({
        "output_dir": str(tmp_path), "figure_size": (3, 3), "dpi": 40,
        "organize_by_type": False, "include_timestamps": False,
    })
    try:
        yield tmp_path
    finally:
        set_config(original)


@pytest.mark.parametrize("radius,count", [(0, 1), (1, 13), (2, 55), (3, 147), (4, 309)])
def test_canonical_shells_match_independent_cartesian_oracle(radius, count):
    sites = enumerate_ivm_sites(radius)
    tuples = [site.as_tuple() for site in sites]
    assert len(tuples) == len(set(tuples)) == count
    assert {site.to_xyz() for site in sites} == _fcc_xyz(radius)
    assert all(min(site) == 0 and sum(site) % 4 == 0 for site in tuples)
    assert tuples == [site.as_tuple() for site in sorted(sites, key=lambda site: (_shell(site), site.as_tuple()))]
    assert tuples == [site.as_tuple() for site in enumerate_ivm_sites(radius)]
    if radius:
        previous = [site.as_tuple() for site in enumerate_ivm_sites(radius - 1)]
        assert tuples[:len(previous)] == previous


@pytest.mark.parametrize("radius,error", [(-1, ValueError), (1.5, TypeError)])
def test_graph_radius_rejects_invalid_sizes(radius, error):
    with pytest.raises(error):
        enumerate_ivm_sites(radius)


def test_fcc_enumeration_does_not_restrict_full_image_nearest_conversion():
    for value in (1, 2**80 + 1, -(2**80 + 1)):
        expected = (value, 0, 0, 0) if value > 0 else (0, -value, -value, -value)
        nearest = QuadrayCoordinate.from_xyz(value, value, value)
        assert nearest.as_tuple() == expected
        assert sum(expected) % 4 != 0
    assert (1, 0, 0, 0) not in {site.as_tuple() for site in enumerate_ivm_sites(3)}


def _plotly_lattice_xyz(path):
    """Read the actual interactive artifact, including Plotly's typed arrays."""
    html = Path(path).read_text()
    call = html.index("Plotly.newPlot(")
    data_start = html.index("[", call)
    traces, _ = json.JSONDecoder().raw_decode(html[data_start:])
    lattice = next(trace for trace in traces if trace.get("name") == "IVM Lattice")

    def values(axis):
        data = lattice[axis]
        if isinstance(data, dict):
            return np.frombuffer(base64.b64decode(data["bdata"]), dtype=data["dtype"])
        return data

    return list(zip(*(values(axis) for axis in "xyz")))


@pytest.mark.parametrize("backend", ["ascii", "matplotlib", "plotly"])
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_real_lattice_backends_agree_with_shell_oracle(output_config, backend, radius):
    if backend == "matplotlib":
        plt = pytest.importorskip("matplotlib.pyplot")
        figures = plt.get_fignums()
    elif backend == "plotly":
        pytest.importorskip("plotly")
    result = plot_ivm_lattice(size=radius, backend=backend)
    expected = _fcc_xyz(radius)
    assert result["metadata"]["total_points"] == len(expected)
    path = Path(result["files"][0])
    assert path.stat().st_size > 0
    if backend == "ascii":
        assert f"Total lattice points: {len(expected)}" in path.read_text()
    elif backend == "plotly":
        xyz = _plotly_lattice_xyz(path)
        assert len(xyz) == len(set(xyz)) == len(expected)
        assert set(xyz) == expected
    else:
        image = plt.imread(path)
        assert image.std() > 0
        assert plt.get_fignums() == figures


@pytest.mark.parametrize("backend", ["ascii", "plotly"])
def test_coordinate_background_is_the_same_origin_centered_fcc_shell(output_config, backend):
    if backend == "plotly":
        pytest.importorskip("plotly")
    # A raw gauge alias far from the coefficient origin must not alter the shell.
    coord = QuadrayCoordinate(101, 100, 100, 100, normalize=False)
    result = plot_quadray_coordinate(coord, lattice_size=1, backend=backend)
    path = Path(result["files"][0])
    if backend == "plotly":
        xyz = _plotly_lattice_xyz(path)
    else:
        coords = [ast.literal_eval(line.strip().split(" <--")[0])
                  for line in path.read_text().splitlines() if line.startswith("  (")]
        assert all(min(q) == 0 and sum(q) % 4 == 0 for q in coords)
        xyz = [QuadrayCoordinate(*q).to_xyz() for q in coords]
    assert len(xyz) == len(set(xyz)) == 13
    assert set(xyz) == _fcc_xyz(1)


def test_demo_lattice_and_spatial_density_count_unique_physical_sites(output_config, capsys):
    from examples.coordinate_systems_demo import demonstrate_ivm_lattice, demonstrate_spatial_patterns

    demonstrate_ivm_lattice()
    lattice_output = capsys.readouterr().out
    for radius in range(1, 5):
        assert f"Graph-shell radius {radius}: {len(_fcc_xyz(radius))} lattice points" in lattice_output
    demonstrate_spatial_patterns()
    density_output = capsys.readouterr().out
    for radius in (1, 2, 3):
        count = sum(sum(v * v for v in xyz) <= radius**2 for xyz in _fcc_xyz(radius))
        density = count / ((4 / 3) * math.pi * radius**3)
        assert f"Radius {radius}: {count} points, density = {density:.3f} points/unit³" in density_output


@pytest.mark.parametrize("backend", ["ascii", "matplotlib", "plotly"])
@pytest.mark.parametrize("transform,error", [
    (lambda q: QuadrayCoordinate.from_xyz(float("nan"), 0, 0), ValueError),
    (lambda q: QuadrayCoordinate(None, 0, 0, 0), TypeError),
    (lambda q: QuadrayCoordinate(Fraction(1, 0), 0, 0, 0), ZeroDivisionError),
])
def test_real_transform_validation_errors_escape_without_artifacts(output_config, backend, transform, error):
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = plt.get_fignums()
    with pytest.raises(error):
        plot_coordinate_transformation(QuadrayCoordinate(0, 0, 0, 0), transform, steps=2, backend=backend)
    assert not list(output_config.rglob("*"))
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("backend", ["ascii", "matplotlib", "plotly"])
def test_later_transform_failure_does_not_save_a_partial_success(output_config, backend):
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = plt.get_fignums()
    limit_error = RuntimeError("translation exceeds the supplied trajectory domain")
    visited = []

    def transform(q):
        visited.append(q.as_tuple())
        if q.a >= 2:
            raise limit_error
        return QuadrayCoordinate(q.a + 1, q.b, q.c, q.d)

    with pytest.raises(RuntimeError) as caught:
        plot_coordinate_transformation(QuadrayCoordinate(0, 0, 0, 0), transform, steps=3, backend=backend)
    assert caught.value is limit_error
    assert visited == [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]
    assert not list(output_config.rglob("*"))
    assert plt.get_fignums() == figures


def test_unexpected_transform_failure_closes_any_figure(output_config):
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = plt.get_fignums()

    def transform(q):
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError):
        plot_coordinate_transformation(QuadrayCoordinate(0, 0, 0, 0), transform,
                                       steps=1, backend="matplotlib")
    assert plt.get_fignums() == figures
    assert not list(output_config.rglob("*"))


def test_transformation_save_failure_closes_real_figure(output_config):
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = plt.get_fignums()
    # A real filesystem collision, not an injected renderer exception.
    (output_config / "coordinate_transformation_2_steps.png").mkdir()
    with pytest.raises(OSError):
        plot_coordinate_transformation(QuadrayCoordinate(0, 0, 0, 0), lambda q: q,
                                       steps=2, backend="matplotlib")
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("kind,filename", [
    ("coordinate", "quadray_coordinate_1_0_0_0.png"),
    ("lattice", "ivm_lattice_size_0.png"),
])
def test_geometry_save_failure_closes_owned_figure_not_callers(output_config, kind, filename):
    plt = pytest.importorskip("matplotlib.pyplot")
    caller = plt.figure()
    try:
        figures = plt.get_fignums()
        (output_config / filename).mkdir()
        with pytest.raises(OSError):
            if kind == "coordinate":
                plot_quadray_coordinate(QuadrayCoordinate(1, 0, 0, 0),
                                       lattice_size=0, backend="matplotlib")
            else:
                plot_ivm_lattice(0, backend="matplotlib")
        assert plt.get_fignums() == figures
        assert plt.fignum_exists(caller.number)
    finally:
        plt.close(caller)


@pytest.mark.parametrize("backend", ["ascii", "matplotlib", "plotly"])
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_transformation_counts_and_data_match_successive_translation(output_config, monkeypatch, backend, steps):
    plt = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.figure import Figure

    initial = QuadrayCoordinate(1, 0, 0, 0)
    displacement = QuadrayCoordinate(0, 1, 0, 0)
    visited = []
    rendered = {}
    savefig = Figure.savefig

    def capture_and_save(fig, *args, **kwargs):
        ax = fig.axes[0]
        rendered["lines"] = [np.column_stack(line.get_data_3d()) for line in ax.lines]
        rendered["markers"] = [np.column_stack(collection._offsets3d)
                               for collection in ax.collections if hasattr(collection, "_offsets3d")]
        return savefig(fig, *args, **kwargs)

    def translate(coord):
        visited.append(coord.as_tuple())
        return coord.add(displacement)

    monkeypatch.setattr(Figure, "savefig", capture_and_save)
    figures = plt.get_fignums()
    result = plot_coordinate_transformation(initial, translate, steps=steps, backend=backend)
    # Adding the B basis changes scale-one XYZ by (-1, 1, -1) per application.
    expected_xyz = np.array([(1 - n, 1 + n, 1 - n) for n in range(steps + 1)])
    assert visited == [(1, n, 0, 0) for n in range(steps)]
    metadata = result["metadata"]
    assert metadata["steps"] == len(visited)
    assert metadata["samples"] == len(expected_xyz)
    end = metadata["end_coordinate"] if backend == "matplotlib" else metadata["end"]
    assert end == (1, steps, 0, 0)
    if backend == "matplotlib":
        if steps:
            np.testing.assert_array_equal(rendered["lines"][0], expected_xyz)
        else:
            assert rendered["lines"] == []
        np.testing.assert_array_equal(rendered["markers"][0], expected_xyz[:1])
        if steps:
            np.testing.assert_array_equal(rendered["markers"][-1], expected_xyz[-1:])
        assert plt.imread(result["files"][0]).std() > 0
    else:
        assert result["files"] == []
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("backend", ["typo", "", "unsupported"])
def test_unknown_transformation_backend_is_rejected_before_callback(output_config, backend):
    def transform(q):
        raise AssertionError("unsupported backend must not execute the callback")

    with pytest.raises(ValueError):
        plot_coordinate_transformation(QuadrayCoordinate(0, 0, 0, 0), transform, backend=backend)
    assert not list(output_config.rglob("*"))


@pytest.mark.parametrize("kind", ["coordinate", "lattice"])
def test_plotly_global_context_fits_every_actual_point_and_basis_endpoint(output_config, kind):
    pytest.importorskip("plotly")
    far = QuadrayCoordinate(100, 0, 0, 0)
    result = (
        plot_quadray_coordinate(far, lattice_size=1, backend="plotly")
        if kind == "coordinate" else
        plot_ivm_lattice(1, highlight_coordinates=[far], backend="plotly")
    )
    html = Path(result["files"][0]).read_text()
    start = html.index("[", html.index("Plotly.newPlot("))
    decoder = json.JSONDecoder()
    traces, end = decoder.raw_decode(html[start:])
    layout, _ = decoder.raw_decode(html[start + end:].lstrip(" ,\n"))

    def values(value):
        if isinstance(value, dict):
            return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        return np.asarray(value)

    points = np.vstack([np.column_stack([values(trace[axis]) for axis in "xyz"])
                        for trace in traces])
    assert any(np.array_equal(point, [100, 100, 100]) for point in points)
    assert set(_plotly_lattice_xyz(result["files"][0])) == _fcc_xyz(1)
    spans = []
    for i, axis in enumerate("xyz"):
        low, high = layout["scene"][f"{axis}axis"]["range"]
        assert low < points[:, i].min() <= points[:, i].max() < high
        assert (low + high) / 2 == pytest.approx((points[:, i].min() + points[:, i].max()) / 2)
        spans.append(high - low)
    assert len(set(spans)) == 1
    assert layout["scene"]["aspectmode"] == "cube"
