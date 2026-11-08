"""Numerical meaning and resource ownership of the scientific poster."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from symergetics.visualization.advanced import (
    _create_decimal_to_symbolic_panel,
    _create_frequency_ratios_panel,
    _create_mathematical_patterns_panel,
    _create_palindrome_analysis_panel,
    _create_pattern_discovery_summary_panel,
    _create_primorial_distribution_panel,
    create_mega_graphical_abstract,
)


@pytest.fixture
def axes():
    figure, ax = plt.subplots()
    try:
        yield ax
    finally:
        plt.close(figure)


def test_approximation_chart_uses_real_bounded_rational_errors(axes):
    _create_decimal_to_symbolic_panel(axes)
    actual = [bar.get_height() for bar in axes.child_axes[0].patches]
    expected = [
        abs(355 / 113 - math.pi),
        abs(1457 / 536 - math.e),
        abs(1393 / 985 - math.sqrt(2)),
        abs(1597 / 987 - (1 + math.sqrt(5)) / 2),
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_dilation_curves_measure_one_shape_in_consistent_units(axes):
    _create_frequency_ratios_panel(axes)
    edge, area, volume = axes.child_axes[0].lines
    for exponent, line in enumerate((edge, area, volume), 1):
        np.testing.assert_allclose(line.get_ydata(), line.get_xdata() ** exponent)


def test_fibonacci_chart_uses_actual_consecutive_integer_ratios(axes):
    _create_mathematical_patterns_panel(axes)
    expected = [1, 2, 3 / 2, 5 / 3, 8 / 5, 13 / 8, 21 / 13,
                34 / 21, 55 / 34, 89 / 55, 144 / 89, 233 / 144,
                377 / 233, 610 / 377]
    np.testing.assert_allclose(axes.child_axes[0].lines[0].get_ydata(), expected)


def test_odd_and_even_palindrome_profiles_share_the_actual_midpoint(axes):
    _create_palindrome_analysis_panel(axes)
    profiles = axes.child_axes[0].lines
    np.testing.assert_array_equal(profiles[0].get_xdata(), [-1, 0, 1])
    np.testing.assert_array_equal(profiles[0].get_ydata(), [1, 2, 1])
    np.testing.assert_array_equal(profiles[3].get_xdata(), [-1.5, -0.5, 0.5, 1.5])
    for profile in profiles[:4]:
        np.testing.assert_array_equal(profile.get_xdata(), -profile.get_xdata()[::-1])
        np.testing.assert_array_equal(profile.get_ydata(), profile.get_ydata()[::-1])


def test_primorial_chart_uses_prime_bounds_not_count_bounds(axes):
    _create_primorial_distribution_panel(axes)
    line = axes.child_axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    np.testing.assert_array_equal(line.get_ydata(),
                                  [2, 6, 30, 210, 2310, 30030, 510510,
                                   9699690, 223092870, 6469693230])


def test_finite_population_counts_exclude_zero_and_leading_zero_strings(axes):
    _create_pattern_discovery_summary_panel(axes)
    counts = [bar.get_height() for bar in axes.child_axes[0].patches]
    np.testing.assert_array_equal(counts, [9, 9, 90])


def test_poster_releases_owned_figure_after_rendering(tmp_path):
    before = plt.get_fignums()
    create_mega_graphical_abstract(output_path=tmp_path / 'poster.svg')
    assert plt.get_fignums() == before


def test_poster_releases_owned_figure_after_save_failure(tmp_path):
    destination = tmp_path / 'blocked.svg'
    destination.mkdir()
    before = plt.get_fignums()
    with pytest.raises(IsADirectoryError):
        create_mega_graphical_abstract(output_path=destination)
    assert plt.get_fignums() == before


def test_poster_rejects_an_unsupported_backend():
    with pytest.raises(ValueError):
        create_mega_graphical_abstract(backend='invalid')
