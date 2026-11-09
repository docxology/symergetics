"""Consumer-visible mathematical data and artifacts from real rendering backends."""

import json
import math
from collections import Counter
from itertools import product
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
    plot_ssrcd_analysis,
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
def test_complex_pattern_renders_endpoint_relations_and_frequency(rendering_output, backend):
    sequence = "12112321123432112345432112345654321"
    result = plot_pattern_analysis(sequence, pattern_type="palindrome", backend=backend)
    counts = Counter(map(int, sequence))
    assert result["metadata"]["length"] == len(sequence)
    assert result["metadata"]["unique_digits"] == len(counts)
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        digits = list(map(int, sequence))
        assert traces["Sequence"]["y"] == digits
        heatmap = next(trace for trace in traces.values() if trace["type"] == "heatmap")
        assert heatmap["z"] == _palindrome_oracle(sequence)[1]
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
        assert radar["r"][:-1] == pytest.approx(list(result["metadata"]["radar_metrics"].values()))
        assert result["metadata"]["radar_metrics"]["normalized_entropy"] == pytest.approx(entropy / math.log2(10))
        assert result["metadata"]["radar_metrics"]["symmetry_score"] == symmetry
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


@pytest.fixture
def saved_matplotlib_figures(monkeypatch):
    """Observe real saved figures without replacing rendering or file output."""
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        result = original(figure, *args, **kwargs)
        saved.append(figure)
        return result

    monkeypatch.setattr(Figure, "savefig", save)
    return saved


def _palindrome_oracle(sequence):
    """Brute-force substrings, independent of the renderer's center expansion."""
    coverage = [0] * len(sequence)
    endpoints = [[0] * len(sequence) for _ in sequence]
    for start in range(len(sequence)):
        for stop in range(start + 2, len(sequence) + 1):
            substring = sequence[start:stop]
            if substring == substring[::-1]:
                coverage[start:stop] = [1] * (stop - start)
                endpoints[start][stop - 1] = endpoints[stop - 1][start] = 1
    return coverage, endpoints


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
@pytest.mark.parametrize("number,source,target,value,result,coefficients", [
    (10, 2, 10, 2, "2", [2]),
    (101, 2, 16, 5, "5", [5]),
    (0, 10, 2, 0, "0", [0]),
    (-10, 2, 10, -2, "-2", [2]),
    ("-aF", 16, 2, -175, "-10101111", [1, 0, 1, 0, 1, 1, 1, 1]),
])
def test_conversion_coefficients_reconstruct_decoded_value(
    rendering_output, saved_matplotlib_figures, backend, number, source, target, value, result, coefficients,
):
    rendered = plot_base_conversion(number, source, target, backend=backend)
    data = rendered["metadata"]
    assert data["result"] == result
    assert data["decimal_value"] == value
    assert data["coefficients"] == coefficients
    assert data["sign"] * sum(c * target ** p for c, p in zip(coefficients, data["powers"])) == value
    for dividend, quotient, remainder in data["division_steps"]:
        assert dividend == quotient * target + remainder
        assert 0 <= remainder < target
    if backend != "ascii":
        traces = _rendered_data(rendered, rendering_output, backend)
        if traces is not None:
            assert traces["Coefficients"]["y"] == coefficients
        else:
            assert [bar.get_height() for bar in saved_matplotlib_figures[-1].axes[2].patches] == coefficients


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
@pytest.mark.parametrize("sequence", ["12341", "1221", "1001", "121", "0110", "0000", "5", "12321"])
def test_contiguous_odd_even_palindrome_coverage(rendering_output, saved_matplotlib_figures, backend, sequence):
    result = plot_pattern_analysis(sequence, "palindrome", backend=backend, include_matrix=True)
    coverage, endpoints = _palindrome_oracle(sequence)
    assert result["metadata"]["number"] == sequence
    assert result["metadata"]["coverage"] == coverage
    assert result["metadata"]["pattern_matrix"] == endpoints
    if backend != "ascii":
        traces = _rendered_data(result, rendering_output, backend)
        if traces is not None:
            heatmap = next(trace for trace in traces.values() if trace["type"] == "heatmap")
            assert heatmap["z"] == endpoints
        else:
            from matplotlib.colors import to_rgba

            axes = saved_matplotlib_figures[-1].axes
            assert axes[2].images[0].get_array().tolist() == endpoints
            colors = get_config()["colors"]
            assert [bar.get_facecolor()[:3] for bar in axes[1].patches] == [
                to_rgba(colors["accent"] if match else colors["primary"])[:3]
                for match in coverage
            ]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
@pytest.mark.parametrize("mode,coverage,pairs", [
    ("repeated", [1, 0, 0, 0, 1], {(0, 4), (4, 0)}),
    ("symmetric", [1, 0, 0, 0, 1], {(0, 4), (4, 0)}),
    ("palindrome", [0, 0, 0, 0, 0], set()),
])
def test_pattern_modes_have_explicit_distinct_relationships(rendering_output, backend, mode, coverage, pairs):
    result = plot_pattern_analysis("12341", mode, backend=backend, include_matrix=True)
    assert result["metadata"]["coverage"] == coverage
    matrix = result["metadata"]["pattern_matrix"]
    assert {(i, j) for i, row in enumerate(matrix) for j, value in enumerate(row) if value} == pairs


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
def test_repeated_is_not_whole_sequence_symmetry(rendering_output, backend):
    repeated = plot_pattern_analysis("1123", "repeated", backend=backend)
    symmetric = plot_pattern_analysis("1123", "symmetric", backend=backend)
    assert repeated["metadata"]["coverage"] == [1, 1, 0, 0]
    assert symmetric["metadata"]["coverage"] == [0, 0, 0, 0]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
def test_ssrcd_even_palindrome_keeps_core_density_distinct(rendering_output, saved_matplotlib_figures, backend):
    from symergetics.computation.palindromes import analyze_scheherazade_ssrcd

    scientific = analyze_scheherazade_ssrcd(1)
    assert scientific["number_string"] == "1001"
    result = plot_ssrcd_analysis(1, backend=backend)
    assert result["metadata"]["palindromic_density"] == scientific["palindromic_density"] == 0
    assert result["metadata"]["palindrome_coverage"] == [1, 1, 1, 1]
    assert result["metadata"]["palindrome_coverage_fraction"] == 1
    assert result["metadata"]["symmetry_score"] == 1
    if backend == "matplotlib":
        assert [bar.get_height() for bar in saved_matplotlib_figures[-1].axes[1].patches] == [1, 1, 1, 1]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("sequence", ["0110", "0000", "0123456789" * 10])
def test_radar_raw_sequence_metrics_are_bounded_fractions(rendering_output, saved_matplotlib_figures, backend, sequence):
    result = plot_pattern_analysis_radar(sequence, backend=backend)
    data = result["metadata"]
    counts = Counter(sequence)
    bigrams = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    pair_counts = Counter(bigrams)
    expected_entropy = -math.fsum(c / len(sequence) * math.log2(c / len(sequence)) for c in counts.values())
    metrics = data["radar_metrics"]
    assert data["sequence"] == sequence
    assert data["sequence_length"] == len(sequence)
    assert data["digit_entropy"] == pytest.approx(expected_entropy)
    assert metrics["repeated_bigram_fraction"] == sum(pair_counts[p] > 1 for p in bigrams) / len(bigrams)
    assert metrics["palindrome_coverage"] == sum(_palindrome_oracle(sequence)[0]) / len(sequence)
    assert metrics["symmetry_score"] == sum(a == b for a, b in zip(sequence[:len(sequence) // 2], sequence[::-1])) / (len(sequence) // 2)
    assert all(0 <= value <= 1 for value in metrics.values())
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        assert traces["Pattern Metrics"]["r"][:-1] == pytest.approx(list(metrics.values()))
    else:
        axis = saved_matplotlib_figures[-1].axes[0]
        assert axis.lines[0].get_ydata()[:-1] == pytest.approx(list(metrics.values()))
        assert axis.get_ylim() == (0, 1)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
@pytest.mark.parametrize("number,source,target", [(12, 2, 10), ("G", 16, 10), (10, 1, 2)])
def test_invalid_conversion_fails_before_artifacts(rendering_output, backend, number, source, target):
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_base_conversion(number, source, target, backend=backend)
    assert plt.get_fignums() == figures
    assert not list(rendering_output.rglob("*"))


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "ascii"])
@pytest.mark.parametrize("number,mode", [("1221", "digit_sequence"), ("", "palindrome"), ("-121", "repeated"), ("12x", "symmetric")])
def test_invalid_pattern_fails_before_artifacts(rendering_output, backend, number, mode):
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_pattern_analysis(number, mode, backend=backend)
    assert plt.get_fignums() == figures
    assert not list(rendering_output.rglob("*"))


@pytest.mark.parametrize("renderer,args", [
    (plot_continued_fraction, (3.25,)),
    (plot_base_conversion, (10, 2, 10)),
    (plot_pattern_analysis, ("1221",)),
    (plot_ssrcd_analysis, (1,)),
    (plot_continued_fraction_convergence, (3.25,)),
    (plot_base_conversion_matrix, (2, 4, 10)),
    (plot_pattern_analysis_radar, ("0110",)),
])
def test_real_save_failure_preserves_callers_and_closes_owned_figures(rendering_output, tmp_path, renderer, args):
    blocked = tmp_path / "regular-file"
    blocked.write_text("not a directory")
    set_config({"output_dir": str(blocked / "renders")})
    caller = plt.figure()
    figures = plt.get_fignums()
    try:
        with pytest.raises(OSError):
            renderer(*args, backend="matplotlib")
        assert plt.get_fignums() == figures
        assert plt.fignum_exists(caller.number)
    finally:
        plt.close(caller)


def test_analysis_failure_after_allocation_closes_owned_ssrcd_figure(rendering_output):
    from symergetics.visualization.mathematical import _plot_ssrcd_matplotlib

    caller = plt.figure()
    figures = plt.get_fignums()
    try:
        with pytest.raises(KeyError):
            _plot_ssrcd_matplotlib({})
        assert plt.get_fignums() == figures
    finally:
        plt.close(caller)


def test_exhaustive_raw_string_scalar_and_matrix_palindromes(rendering_output):
    from symergetics.visualization.mathematical import _radar_data

    for length in range(1, 7):
        for digits in product("01", repeat=length):
            sequence = "".join(digits)
            coverage, endpoints = _palindrome_oracle(sequence)
            scalar = plot_pattern_analysis(sequence, backend="ascii", include_matrix=False)
            dense = plot_pattern_analysis(sequence, backend="ascii")
            assert scalar["metadata"]["number"] == dense["metadata"]["number"] == sequence
            assert scalar["metadata"]["coverage"] == dense["metadata"]["coverage"] == coverage
            assert scalar["metadata"]["pattern_matrix"] is None
            assert dense["metadata"]["pattern_matrix"] == endpoints
            assert _radar_data(sequence)["radar_metrics"]["palindrome_coverage"] == sum(coverage) / length


@pytest.mark.parametrize("mode", ["repeated", "symmetric"])
@pytest.mark.parametrize("sequence", ["0", "0000", "0110", "00123", "01210"])
def test_ascii_membership_is_independent_of_matrix_materialization(rendering_output, mode, sequence):
    scalar = plot_pattern_analysis(sequence, mode, backend="ascii", include_matrix=False)
    dense = plot_pattern_analysis(sequence, mode, backend="ascii")
    if mode == "repeated":
        expected = [int(sequence.count(digit) > 1) for digit in sequence]
    else:
        expected = [int(i != len(sequence) - 1 - i and digit == sequence[-1 - i])
                    for i, digit in enumerate(sequence)]
    assert scalar["metadata"]["coverage"] == dense["metadata"]["coverage"] == expected
    assert scalar["metadata"]["pattern_matrix"] is None
    matrix = dense["metadata"]["pattern_matrix"]
    assert [int(any(row)) for row in matrix] == expected


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize("number,source", [(0, 10), (10, 10), ("-aF", 16), ("1010", 2)])
def test_binary_rendered_weights_reconstruct_decoded_value(
    rendering_output, saved_matplotlib_figures, backend, number, source,
):
    result = plot_base_conversion(number, source, 2, backend=backend)
    data = result["metadata"]
    traces = _rendered_data(result, rendering_output, backend)
    if traces is not None:
        coefficients = traces["Coefficients"]["y"]
        weights = [trace for trace in traces.values() if trace.get("xaxis") == "x4"][0]["y"]
    else:
        figure = saved_matplotlib_figures[-1]
        coefficients = [bar.get_height() for bar in figure.axes[2].patches]
        weights = [bar.get_height() for bar in figure.axes[3].patches]
    assert weights == [2 ** exponent for exponent in data["powers"]]
    assert data["sign"] * sum(c * weight for c, weight in zip(coefficients, weights)) == data["decimal_value"]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_graphical_matrix_cannot_be_omitted_without_losing_displayed_data(rendering_output, backend):
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_pattern_analysis("0110", backend=backend, include_matrix=False)
    assert plt.get_fignums() == figures
    assert not list(rendering_output.rglob("*"))
