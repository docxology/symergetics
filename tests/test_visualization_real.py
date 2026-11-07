"""Exercise real visualization APIs and their configured output containment."""

from pathlib import Path

from symergetics.core.constants import SymergeticsConstants
from symergetics.core.numbers import SymergeticsNumber
from symergetics.visualization import get_config, set_config
from symergetics.visualization.advanced import create_comparative_analysis_visualization
from symergetics.visualization.geometry import (
    plot_ivm_lattice,
    plot_polyhedron,
    plot_quadray_coordinate,
)
from symergetics.visualization.mathematical import (
    plot_base_conversion,
    plot_continued_fraction,
    plot_pattern_analysis,
    plot_ssrcd_analysis,
)
from symergetics.visualization.numbers import (
    plot_mnemonic_visualization,
    plot_palindromic_pattern,
    plot_primorial_distribution,
    plot_scheherazade_pattern,
)


def assert_output_contained(result):
    """Check actual files, not ignored output_dir keyword arguments."""
    output_dir = Path(get_config()['output_dir']).resolve()
    assert result['files']
    assert result['metadata']
    for filename in result['files']:
        path = Path(filename).resolve()
        assert path.is_relative_to(output_dir)
        assert path.is_file()
        assert path.stat().st_size > 0


class TestVisualizationNumbers:
    def test_plot_palindromic_pattern_basic(self):
        assert_output_contained(plot_palindromic_pattern(12321, backend='ascii'))

    def test_plot_palindromic_pattern_symergetics_number(self):
        assert_output_contained(plot_palindromic_pattern(SymergeticsNumber(12321), backend='ascii'))

    def test_plot_scheherazade_pattern(self):
        assert_output_contained(plot_scheherazade_pattern(3, backend='ascii'))

    def test_plot_primorial_distribution(self):
        assert_output_contained(plot_primorial_distribution(max_n=10, backend='ascii'))

    def test_plot_mnemonic_visualization(self):
        assert_output_contained(plot_mnemonic_visualization(SymergeticsNumber(12345), backend='ascii'))


class TestVisualizationGeometry:
    def test_plot_polyhedron_tetrahedron(self):
        assert_output_contained(plot_polyhedron('tetrahedron', backend='ascii'))

    def test_plot_polyhedron_octahedron(self):
        assert_output_contained(plot_polyhedron('octahedron', backend='ascii'))

    def test_plot_quadray_coordinate(self):
        from symergetics.core.coordinates import QuadrayCoordinate

        assert_output_contained(plot_quadray_coordinate(QuadrayCoordinate(1, 2, 3, 4), backend='ascii'))

    def test_plot_ivm_lattice(self):
        assert_output_contained(plot_ivm_lattice(size=3, backend='ascii'))


class TestVisualizationMathematical:
    def test_plot_continued_fraction(self):
        assert_output_contained(plot_continued_fraction(value=3.14159, backend='ascii'))

    def test_plot_base_conversion(self):
        assert_output_contained(plot_base_conversion(number=12345, backend='ascii'))

    def test_plot_pattern_analysis(self):
        assert_output_contained(plot_pattern_analysis(number=12321, backend='ascii'))

    def test_plot_ssrcd_analysis(self):
        assert_output_contained(plot_ssrcd_analysis(power=3, backend='ascii'))


class TestVisualizationAdvanced:
    def test_create_comparative_analysis_visualization(self):
        result = create_comparative_analysis_visualization(
            domain1_data=[{'number': 12321, 'is_palindromic': True}],
            domain2_data=[{'number': 12345, 'is_palindromic': False}],
            domain1_name='Palindromes',
            domain2_name='Non-palindromes',
            backend='matplotlib',
        )
        assert_output_contained(result)


class TestVisualizationIntegration:
    def test_visualization_with_constants(self):
        tetra_vol = SymergeticsConstants().get_volume_ratio('tetrahedron')
        assert_output_contained(plot_palindromic_pattern(tetra_vol, backend='ascii'))

    def test_visualization_with_coordinates(self):
        from symergetics.core.coordinates import QuadrayCoordinate

        assert_output_contained(plot_quadray_coordinate(QuadrayCoordinate(1, 2, 3, 4), backend='ascii'))

    def test_visualization_error_handling(self):
        result = plot_palindromic_pattern('invalid', backend='ascii')
        assert result['files'] == []
        assert result['metadata']['error']


class TestVisualizationPerformance:
    def test_large_number_visualization(self):
        assert_output_contained(plot_palindromic_pattern(SymergeticsNumber(10**20), backend='ascii'))

    def test_ascii_output_format(self):
        result = plot_scheherazade_pattern(2, backend='ascii')
        assert_output_contained(result)
        assert all(Path(filename).suffix == '.txt' for filename in result['files'])

    def test_custom_configuration(self, tmp_path):
        output_dir = tmp_path / 'custom-output'
        set_config({
            'output_dir': str(output_dir),
            'figure_size': (12, 10),
            'colors': {'primary': '#ff0000'},
            'dpi': 150,
        })
        result = plot_primorial_distribution(max_n=5, backend='ascii')
        assert_output_contained(result)
        assert get_config()['figure_size'] == (12, 10)
        assert get_config()['colors']['primary'] == '#ff0000'
        assert get_config()['dpi'] == 150
