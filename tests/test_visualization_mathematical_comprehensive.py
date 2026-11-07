"""Consumer-visible mathematical data and artifacts from real rendering backends."""

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from symergetics.visualization import get_config, set_config
from symergetics.visualization.mathematical import (
    plot_base_conversion,
    plot_base_conversion_matrix,
    plot_continued_fraction,
    plot_continued_fraction_convergence,
    plot_pattern_analysis,
    plot_pattern_analysis_radar,
)


@pytest.fixture
def rendering_output(tmp_path):
    """Own configuration locally; conftest also contains default outputs in cwd."""
    original = get_config()
    figures = plt.get_fignums()
    output = tmp_path / "renders"
    set_config({
        "output_dir": str(output), "figure_size": (6, 5), "dpi": 40,
        "organize_by_type": False, "include_timestamps": False,
    })
    try:
        yield output
        assert plt.get_fignums() == figures
    finally:
        for figure in set(plt.get_fignums()) - set(figures):
            plt.close(figure)
        set_config(original)


def _rendered_data(result, output, backend):
    """Decode the actual PNG or interactive HTML, never a mocked figure."""
    assert result["metadata"]["backend"] == backend
    assert len(result["files"]) == 1
    path = Path(result["files"][0])
    assert path.resolve().is_relative_to(output.resolve())
    assert path.stat().st_size > 0
    if backend == "matplotlib":
        assert path.suffix == ".png"
        image = plt.imread(path)
        assert image.shape[0] > 1 and image.shape[1] > 1
        assert image[..., :3].std() > 0
        return None
    assert path.suffix == ".html"
    html = path.read_text()
    call = html.index("Plotly.newPlot(")
    data_start = html.index("[", call)
    traces, _ = json.JSONDecoder().raw_decode(html[data_start:])
    assert traces
    return {trace.get("name", trace["type"]): trace for trace in traces}


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("value,max_terms,terms,approximations", [
    (5, 10, [5], [5]),
    (math.pi, 3, [3, 7, 15], [3, 22 / 7, 333 / 106]),
    (1e-10, 1000, [0, 10_000_000_000], [0, 1e-10]),
])
def test_convergence_integer_truncated_and_tiny_values(
    rendering_output, backend, value, max_terms, terms, approximations,
):
    result = plot_continued_fraction_convergence(value, max_terms=max_terms, backend=backend)
    assert result["metadata"]["terms_found"] == len(terms)
    assert result["metadata"]["convergents_found"] == len(approximations)
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        assert traces["Terms"]["y"] == terms
        assert traces["Convergents"]["y"] == pytest.approx(approximations)
        assert traces["Error"]["y"] == pytest.approx([abs(a - value) for a in approximations])


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_exact_rational_continued_fraction(rendering_output, backend):
    result = plot_continued_fraction(3.25, max_terms=20, backend=backend)
    assert result["metadata"]["terms_calculated"] == 2
    assert result["metadata"]["convergents_found"] == 2
    assert result["metadata"]["final_error"] == 0
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        assert traces["Terms"]["y"] == [3, 4]
        assert traces["Convergents"]["y"] == [3, 3.25]
        assert traces["Error"]["y"] == [0.25, 0]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_large_number_conversion_uses_requested_base(rendering_output, backend):
    number = 10**9
    result = plot_base_conversion(number, from_base=10, to_base=16, backend=backend)
    expected = format(number, "X")
    assert result["metadata"]["result"] == expected
    assert result["metadata"]["steps"] == len(expected)
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        coefficients = [int(digit, 16) for digit in expected]
        assert traces["Remainders"]["y"] == coefficients
        assert traces["Coefficients"]["y"] == coefficients


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("number", [0, 10**6])
def test_matrix_clamps_to_supported_bases_and_preserves_values(rendering_output, backend, number):
    result = plot_base_conversion_matrix(start_base=1, end_base=37, number=number, backend=backend)
    assert result["metadata"]["total_bases"] == 35
    assert result["metadata"]["valid_conversions"] == 35
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        bases, representations, lengths = traces["table"]["cells"]["values"]
        assert bases == list(range(2, 37))
        assert [int(value, base) for base, value in zip(bases, representations)] == [number] * 35
        assert lengths == [str(len(value)) for value in representations]
        for base, expected in [(2, format(number, "b")), (8, format(number, "o")),
                               (10, str(number)), (16, format(number, "X"))]:
            assert representations[bases.index(base)] == expected


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_complex_pattern_renders_similarity_and_frequency(rendering_output, backend):
    sequence = "12112321123432112345432112345654321"
    result = plot_pattern_analysis(sequence, pattern_type="palindrome", backend=backend)
    counts = Counter(map(int, sequence))
    assert result["metadata"]["length"] == len(sequence)
    assert result["metadata"]["unique_digits"] == len(counts)
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        digits = list(map(int, sequence))
        assert traces["Sequence"]["y"] == digits
        assert traces["Similarity"]["z"] == [[int(a == b) for b in digits] for a in digits]
        assert dict(zip(traces["Frequency"]["x"], traces["Frequency"]["y"])) == counts


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("sequence,entropy,symmetry", [
    ("5", 0, 0),
    ("1221" * 30, 1, 1),
])
def test_radar_singleton_and_long_balanced_palindrome(
    rendering_output, backend, sequence, entropy, symmetry,
):
    result = plot_pattern_analysis_radar(sequence, backend=backend)
    assert result["metadata"]["sequence_length"] == len(sequence)
    assert result["metadata"]["unique_digits"] == len(set(sequence))
    entropy_key = "entropy" if backend == "matplotlib" else "digit_entropy"
    assert result["metadata"][entropy_key] == pytest.approx(entropy)
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        radar = traces["Pattern Metrics"]
        assert radar["r"][0] == min(len(sequence) / 100, 1)
        assert radar["r"][1] == len(set(sequence)) / 10
        assert radar["r"][2] == pytest.approx(entropy / 4)
        assert radar["r"][-2] == symmetry
        assert radar["r"][-1] == radar["r"][0]
        assert radar["theta"][-1] == radar["theta"][0]


@pytest.mark.parametrize("renderer,arguments", [
    (plot_continued_fraction_convergence, {"value": math.pi}),
    (plot_base_conversion_matrix, {"number": 255}),
    (plot_pattern_analysis_radar, {"sequence": "1221"}),
])
@pytest.mark.parametrize("backend", ["ascii", "invalid"])
def test_unsupported_backends_fail_without_artifacts(rendering_output, renderer, arguments, backend):
    with pytest.raises(ValueError):
        renderer(backend=backend, **arguments)
    assert not list(rendering_output.rglob("*"))


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("value", ["invalid", float("nan"), float("inf")])
def test_non_real_or_non_finite_convergence_rejected_without_artifacts(rendering_output, backend, value):
    with pytest.raises((ValueError, OverflowError)):
        plot_continued_fraction_convergence(value, max_terms=15, backend=backend)
    assert not list(rendering_output.rglob("*"))
