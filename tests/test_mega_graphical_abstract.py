"""Numerical meaning and resource ownership of the scientific poster."""
import math
from fractions import Fraction
import json
from xml.etree import ElementTree
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil

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


@pytest.mark.parametrize('section', ['', 'unknown', None, ['geometry']])
def test_invalid_section_fails_before_allocating_a_figure(tmp_path, section):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        create_mega_graphical_abstract(output_path=tmp_path / 'invalid.svg', section=section)
    assert plt.get_fignums() == before
    assert not list(tmp_path.iterdir())


def test_topic_selection_preserves_computed_data_and_owned_resources(tmp_path):
    before = plt.get_fignums()
    complete = create_mega_graphical_abstract(
        output_path=tmp_path / 'all.svg', section='all',
    )
    grouped_data = {}
    for section in ('geometry', 'arithmetic', 'patterns'):
        result = create_mega_graphical_abstract(
            output_path=tmp_path / f'{section}.svg', section=section,
        )
        assert not grouped_data.keys() & result['data'].keys()
        grouped_data.update(result['data'])
        assert plt.get_fignums() == before
    assert set(grouped_data) == {
        'symbolic_operations', 'geometric_ratios', 'mathematical_patterns',
        'decimal_to_sphere_packing', 'frequency_ratios', 'natural_language',
        'sphere_packing', 'ivm_lattice', 'vector_equilibrium',
        'decimal_to_symbolic', 'continued_fractions', 'quadray_coordinates',
        'palindrome_analysis', 'primorial_distribution', 'pattern_discovery_summary',
        'synergetics_sphere_frequency', 'geometry_precision',
    }
    assert grouped_data == complete['data']
    assert complete['metadata']['panel_ids'] == list(complete['data'])
    assert complete['metadata']['panels'] == len(complete['data'])
    json.dumps(complete, allow_nan=False)
    cuboctahedron = complete['data']['vector_equilibrium']
    assert Fraction(cuboctahedron['volume_xyz_cubed_exact']) == (
        Fraction(cuboctahedron['volume_ivm_exact']) * Fraction(8, 3)
    )
    fibonacci = complete['data']['mathematical_patterns']
    for ratio, a, b in zip(fibonacci['consecutive_ratios_exact'],
                           fibonacci['fibonacci_integers'],
                           fibonacci['fibonacci_integers'][1:]):
        assert Fraction(ratio) == Fraction(b, a)
    np.testing.assert_allclose(
        complete['data']['sphere_packing']['contact_distances_xyz'], 2,
    )
    assert plt.get_fignums() == before


def test_explicit_positional_title_is_rendered_when_switching_sections(tmp_path):
    title = 'Caller-selected scientific title'
    destination = tmp_path / 'custom.svg'
    with plt.rc_context({'svg.fonttype': 'none'}):
        result = create_mega_graphical_abstract(
            title, destination, 'matplotlib', section='patterns',
        )
    root = ElementTree.parse(destination).getroot()
    rendered_text = [''.join(node.itertext())
                     for node in root.iter('{http://www.w3.org/2000/svg}text')]
    assert title in rendered_text
    assert result['metadata']['title'] == title


@pytest.mark.parametrize('section', ['geometry', 'arithmetic', 'patterns', 'all'])
def test_selected_poster_releases_figure_after_save_failure(tmp_path, section):
    destination = tmp_path / 'blocked.svg'
    destination.mkdir()
    before = plt.get_fignums()
    with pytest.raises(IsADirectoryError):
        create_mega_graphical_abstract(output_path=destination, section=section)
    assert plt.get_fignums() == before


@pytest.mark.parametrize('epoch, expected', [
    ('946684800', '2000-01-01'),
    ('946681200', '1999-12-31'),
    (None, None),
])
def test_poster_generation_date_uses_source_epoch_or_local_date(
    tmp_path, monkeypatch, epoch, expected,
):
    if epoch is None:
        monkeypatch.delenv('SOURCE_DATE_EPOCH', raising=False)
    else:
        monkeypatch.setenv('SOURCE_DATE_EPOCH', epoch)
    local_before = datetime.now().strftime('%Y-%m-%d')
    destination = tmp_path / 'dated.svg'
    with plt.rc_context({'svg.fonttype': 'none'}):
        create_mega_graphical_abstract(output_path=destination, section='patterns')
    local_after = datetime.now().strftime('%Y-%m-%d')
    root = ElementTree.parse(destination).getroot()
    dates = [
        ''.join(node.itertext()).removeprefix('Generated: ')
        for node in root.iter('{http://www.w3.org/2000/svg}text')
        if ''.join(node.itertext()).startswith('Generated: ')
    ]
    assert len(dates) == 1
    assert dates[0] in ({expected} if expected is not None else {local_before, local_after})


def test_scientific_sidecars_are_portable_without_changing_runtime_paths(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'examples/comprehensive_visualizations_demo.py'
    spec = importlib.util.spec_from_file_location('comprehensive_demo', source)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    destination = tmp_path / 'original'
    results = demo.demonstrate_mega_graphical_abstracts(destination)
    assert len(results) == 5
    relocated = tmp_path / 'relocated'
    shutil.copytree(destination, relocated)
    for result in results:
        assert 'error' not in result
        assert all(Path(filename).is_absolute() for filename in result['files'])
        sidecar = Path(result['files'][-1])
        saved = json.loads(sidecar.read_text())
        assert saved['metadata'] == result['metadata']
        assert saved['data'] == json.loads(json.dumps(result['data'], allow_nan=False))
        assert all(not Path(filename).is_absolute() for filename in saved['files'])
        assert saved['files'] == [Path(filename).name for filename in result['files']]
        for filename in saved['files']:
            assert (relocated / filename).is_file()
            assert (relocated / filename).read_bytes() == (destination / filename).read_bytes()
