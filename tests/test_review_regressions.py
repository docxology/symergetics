"""
Regression tests for review fixes (2026-09-05).

Each test pins a behavior that was broken before the fix:
- SCHEHERAZADE_POWERS drifted from 1001**n
- CUBE_VERTICES / OCTAHEDRON_VERTICES were not a cube / octahedron
- integer_tetra_volume mixed normalization scales (raw det vs det/4)
- SymergeticsNumber.pi was float-derived (~16 digits regardless of arg)
- SymergeticsNumber("0.1") went through a lossy float round-trip
- convert_between_bases hung on base 1 and mishandled negatives
- xyz_to_quadray truncated floats and ignored the lattice phase ambiguity
- from_xyz broke on exact half-integer quadray solutions (banker's rounding)
- continued_fraction_approximation truncated toward zero for negatives
- format_as_mixed_number rendered impossible mixed numbers for negatives
- mnemonic round-trips lost the sign
- next_prime skipped 2
- scheherazade_pascal_coefficients / find_palindromic_sequence were hardcoded
- analyze_binomial_patterns matched on first digits only (false positives)
- visualization __all__ was shadowed by a narrower second __all__
- importing symergetics.visualization required matplotlib at import time
- viz modules read a stale _config dict after set_config/reset_config
"""

import math
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest

from symergetics.core.constants import SymergeticsConstants
from symergetics.core.coordinates import (
    CUBE_VERTICES,
    IVM_NEIGHBORS,
    OCTAHEDRON_VERTICES,
    QuadrayCoordinate,
    urner_embedding,
)
from symergetics.core.numbers import SymergeticsNumber
from symergetics.geometry.polyhedra import (
    Cube,
    Cuboctahedron,
    Octahedron,
    Tetrahedron,
    ace_tetra_volume_5x5,
    integer_tetra_volume,
)
from symergetics.utils.conversion import (
    continued_fraction_approximation,
    convert_between_bases,
    format_as_mixed_number,
    xyz_to_quadray,
)
from symergetics.utils.mnemonics import (
    format_large_number,
    mnemonic_decode,
    mnemonic_encode,
)
from symergetics.visualization import reset_config, set_config


class TestScheherazadePowers:
    """The constants table must equal the actual powers of 1001."""

    def test_table_matches_1001_powers(self):
        for n, value in SymergeticsConstants.SCHEHERAZADE_POWERS.items():
            assert value.value == Fraction(1001) ** n, f"entry {n} drifted"

    def test_get_scheherazade_power_returns_exact_value(self):
        for n in range(1, 7):
            assert SymergeticsConstants.get_scheherazade_power(n).value == Fraction(1001) ** n


class TestPolyhedronGeometry:
    """Vertex constants must form the regular solids they claim to be."""

    @staticmethod
    def _distance_squared_spectrum(vertices):
        points = [v.to_xyz(urner_embedding()) for v in vertices]
        spectrum = sorted(
            sum((a[i] - b[i]) ** 2 for i in range(3))
            for i, a in enumerate(points)
            for b in points[i + 1:]
        )
        return [round(d, 9) for d in spectrum]

    def test_cube_vertices_form_a_cube(self):
        # A cube has 12 edges e, 12 face diagonals e*sqrt(2), 4 space diagonals e*sqrt(3)
        spectrum = self._distance_squared_spectrum(CUBE_VERTICES)
        assert spectrum.count(spectrum[0]) == 12   # edges
        assert spectrum.count(spectrum[0] * 2) == 12   # face diagonals
        assert spectrum.count(spectrum[0] * 3) == 4    # space diagonals

    def test_octahedron_vertices_form_an_octahedron(self):
        # An octahedron has 12 equal edges and 3 equal long diagonals
        spectrum = self._distance_squared_spectrum(OCTAHEDRON_VERTICES)
        assert spectrum.count(spectrum[0]) == 12
        assert spectrum.count(spectrum[0] * 2) == 3
        assert len(spectrum) == 15

    def test_ivm_neighbors_form_cuboctahedron_shell(self):
        assert len(IVM_NEIGHBORS) == 12
        spectrum = self._distance_squared_spectrum(IVM_NEIGHBORS)
        assert spectrum.count(spectrum[0]) == 24  # cuboctahedron edges

    def test_default_volumes_match_ivm_ratios(self):
        assert Tetrahedron().volume() == 1
        assert Octahedron().volume() == 4
        assert Cube().volume() == 3
        assert Cuboctahedron().volume() == 20

    def test_volume_tracks_custom_vertices(self):
        # Double all coordinates: tetravolume scales by 2^3 = 8
        doubled = [QuadrayCoordinate(v.a * 2, v.b * 2, v.c * 2, v.d * 2)
                   for v in OCTAHEDRON_VERTICES]
        assert Octahedron(doubled).volume() == 32

    def test_edge_and_face_counts(self):
        assert len(Tetrahedron().edges()) == 6
        assert len(Octahedron().edges()) == 12
        assert len(Cube().edges()) == 12
        assert len(Cuboctahedron().edges()) == 24
        assert len(Tetrahedron().faces()) == 4
        assert len(Octahedron().faces()) == 8
        assert len(Cube().faces()) == 6
        assert len(Cuboctahedron().faces()) == 14

    def test_ace_and_projection_volumes_agree(self):
        import random

        rng = random.Random(20260905)
        for _ in range(100):
            points = [QuadrayCoordinate(*[rng.randint(-6, 6) for _ in range(4)])
                      for _ in range(4)]
            assert integer_tetra_volume(*points) == ace_tetra_volume_5x5(*points)

    def test_corner_tetrahedron_volume_is_one_quarter(self):
        corner = integer_tetra_volume(
            QuadrayCoordinate(0, 0, 0, 0),
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
            QuadrayCoordinate(0, 0, 1, 0),
        )
        assert corner == Fraction(1, 4)


class TestExactNumbers:
    def test_pi_digits_argument_is_real(self):
        # 50 digits of pi (known constant prefix)
        known = "3.14159265358979323846264338327950288419716939937510"
        value = SymergeticsNumber.pi(digits=50).value
        # decimal expansion
        n, d = value.numerator, value.denominator
        digits = []
        rem = n % d
        for _ in range(60):
            rem *= 10
            digits.append(str(rem // d))
            rem %= d
        expansion = "3." + "".join(digits)
        assert expansion.startswith(known)

    def test_pi_rejects_nonpositive_digits(self):
        with pytest.raises(ValueError):
            SymergeticsNumber.pi(digits=0)

    def test_decimal_string_parses_exactly(self):
        assert SymergeticsNumber("0.1").value == Fraction(1, 10)
        assert SymergeticsNumber("3.14159").value == Fraction(314159, 100000)


class TestBaseConversion:
    def test_decimal_to_base_seven(self):
        assert convert_between_bases(1001, 10, 7) == "2630"

    def test_negative_decimal_to_binary(self):
        assert convert_between_bases(-10, 10, 2) == "-1010"

    def test_negative_binary_to_decimal(self):
        assert convert_between_bases("-101", 2, 10) == "-5"

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            convert_between_bases(5, 10, 1)
        with pytest.raises(ValueError):
            convert_between_bases(5, 10, 37)
        with pytest.raises(ValueError):
            convert_between_bases(5, 0, 10)

    def test_digit_out_of_range_raises(self):
        with pytest.raises(ValueError):
            convert_between_bases(129, 2, 10)  # '9' is not a binary digit

    def test_round_trip_through_arbitrary_base(self):
        for number in (0, 1, 255, 1001, 65535):
            for base in (2, 8, 16, 36):
                rendered = convert_between_bases(number, 10, base)
                assert convert_between_bases(rendered, base, 10) == str(number)


class TestQuadrayConversions:
    def test_lattice_point_round_trip(self):
        coord = xyz_to_quadray(2.0, 0.0, 0.0)
        assert coord.as_tuple() == (1, 0, 0, 1)
        assert coord.to_xyz() == (2.0, 0.0, 0.0)

    def test_random_lattice_points_round_trip(self):
        import random

        rng = random.Random(7)
        for _ in range(200):
            point = [rng.randint(-5, 5) for _ in range(4)]
            original = QuadrayCoordinate(*point)
            recovered = QuadrayCoordinate.from_xyz(*original.to_xyz())
            assert recovered == original

    def test_non_lattice_point_snaps_to_nearest(self):
        # (1, 0, 0) is not an IVM point; the nearest lattice point is the origin
        assert xyz_to_quadray(1.0, 0.0, 0.0).as_tuple() == (0, 0, 0, 0)

    def test_float_coordinates_snap_to_nearest_integer(self):
        assert QuadrayCoordinate(1.9, -1.2, 0, 3.7, normalize=False).as_tuple() == (2, -1, 0, 4)


class TestContinuedFractions:
    def test_negative_value_uses_floor(self):
        assert continued_fraction_approximation(-2.5, 5) == [-3, 2]

    def test_pi_terms(self):
        assert continued_fraction_approximation(3.14159, 4) == [3, 7, 15, 1]

    def test_non_numeric_input_raises_value_error(self):
        with pytest.raises(ValueError):
            continued_fraction_approximation("invalid", 5)


class TestMixedNumbers:
    def test_negative_uses_floor_convention(self):
        assert format_as_mixed_number(Fraction(-7, 3)) == "-3 2/3"

    def test_positive_unchanged(self):
        assert format_as_mixed_number(Fraction(7, 3)) == "2 1/3"


class TestMnemonicRoundTrip:
    def test_negative_grouped_round_trip(self):
        encoded = mnemonic_encode(-1001, style="grouped")
        assert "negative" in encoded
        assert mnemonic_decode(encoded) == -1001

    def test_positive_round_trip(self):
        assert mnemonic_decode(mnemonic_encode(30030, style="grouped")) == 30030

    def test_format_large_number_preserves_sign(self):
        assert format_large_number(-1234) == "-1,234"
        assert format_large_number(-10 ** 20, grouping=4).startswith("-")


class TestPrimorialHelpers:
    def test_next_prime_includes_two(self):
        from symergetics.computation.primorials import next_prime

        assert next_prime(0) == 2
        assert next_prime(1) == 2
        assert next_prime(2) == 3
        assert next_prime(7) == 11

    def test_pascal_coefficients_general(self):
        from symergetics.computation.primorials import scheherazade_pascal_coefficients

        assert scheherazade_pascal_coefficients(6) == [1, 6, 15, 20, 15, 6, 1]
        assert scheherazade_pascal_coefficients(1) == [1, 1]
        assert scheherazade_pascal_coefficients(4) == [1, 4, 6, 4, 1]
        # Power 13: coefficients overflow the 3-digit block -> no clean pattern
        assert scheherazade_pascal_coefficients(13) == []

    def test_find_palindromic_sequence_generalizes(self):
        from symergetics.computation.palindromes import find_palindromic_sequence

        for power in (1, 2, 4, 6):
            analysis = find_palindromic_sequence(power)
            assert analysis["pascal_coefficients"] == [
                math.comb(power, k) for k in range(power + 1)
            ]
            assert int("".join(analysis["ssrcd_patterns"])) == 1001 ** power

    def test_binomial_matcher_has_no_first_char_false_positives(self):
        from symergetics.computation.palindromes import analyze_binomial_patterns

        # '1612161' matches the first characters of row 6 (1,6,1,2,1,6,1)
        # but is NOT the zero-padded concatenation of the row.
        assert analyze_binomial_patterns("1612161") == []
        # The real 1001^6 expansion does contain row 6 (unpadded concatenation)
        hits = analyze_binomial_patterns(str(1001 ** 6))
        assert any(h["description"] == "Pascal's Triangle Row 6" for h in hits)


class TestVisualizationSurface:
    def test_all_names_resolve(self):
        import symergetics.visualization as viz

        missing = [name for name in viz.__all__ if not hasattr(viz, name)]
        assert missing == []

    def test_star_import_exports_everything(self):
        namespace = {}
        exec("from symergetics.visualization import *", namespace)
        import symergetics.visualization as viz

        missing = [name for name in viz.__all__ if name not in namespace]
        assert missing == []

    def test_package_imports_without_matplotlib(self):
        code = (
            "import importlib.abc\n"
            "import sys\n"
            "class Blocker(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'matplotlib' or name.startswith('matplotlib.'):\n"
            "            raise ImportError('blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import symergetics.visualization\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_config_reset_is_visible_to_all_backends(self):
        # Regression: viz modules used to bind the _config dict object, so
        # reset_config() never reached them and a stale 'plotly' backend
        # leaked into matplotlib-only functions.
        import tempfile

        from symergetics.visualization import (
            create_comparative_analysis_visualization,
            reset_config,
            set_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            set_config({"output_dir": tmp})
            try:
                result = create_comparative_analysis_visualization(
                    [{"length": 3, "is_palindromic": True, "palindromic_density": 1.0}],
                    [{"length": 5, "is_palindromic": False, "palindromic_density": 0.0}],
                )
            finally:
                reset_config()
        assert result["metadata"]["type"] == "comparative_analysis"



class TestPlotlyBackends:
    """Real plotly renders (files written, backend reported honestly)."""

    def test_numbers_plotly_backends_render(self):
        from symergetics.visualization.numbers import (
            plot_palindromic_pattern,
            plot_primorial_distribution,
            plot_primorial_spectrum,
            plot_scheherazade_pattern,
        )

        functions = [
            (plot_palindromic_pattern, (12321,), {}),
            (plot_scheherazade_pattern, (2,), {}),
            (plot_primorial_distribution, (8,), {}),
            (plot_primorial_spectrum, (8,), {}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                results = [fn(*args, backend='plotly', **kwargs)
                           for fn, args, kwargs in functions]
            finally:
                reset_config()

            for result in results:
                assert result['metadata']['backend'] == 'plotly'
                for file_path in result['files']:
                    assert os.path.exists(file_path)
                    assert os.path.getsize(file_path) > 0

    def test_mathematical_plotly_backends_render(self):
        from symergetics.visualization.mathematical import (
            plot_base_conversion,
            plot_base_conversion_matrix,
            plot_continued_fraction_convergence,
            plot_pattern_analysis_radar,
        )

        functions = [
            (plot_base_conversion, (1001, 10, 2), {}),
            (plot_base_conversion_matrix, (), {'start_base': 2, 'end_base': 10,
                                               'number': 255}),
            (plot_continued_fraction_convergence, (3.14159, 8), {}),
            (plot_pattern_analysis_radar, ('1234554321',), {}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                results = [fn(*args, backend='plotly', **kwargs)
                           for fn, args, kwargs in functions]
            finally:
                reset_config()

            for result in results:
                assert result['metadata']['backend'] == 'plotly'
                for file_path in result['files']:
                    assert os.path.exists(file_path)
                    assert os.path.getsize(file_path) > 0

    def test_plotly_heatmap_no_seaborn_required(self):
        # The plotly heatmap is pure plotly (matplotlib path needs seaborn)
        from symergetics.visualization.numbers import plot_palindromic_heatmap

        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_palindromic_heatmap(100, 110, backend='plotly')
            finally:
                reset_config()

            assert result['metadata']['backend'] == 'plotly'
            assert result['metadata']['palindromic_count'] > 0
            for file_path in result['files']:
                assert os.path.exists(file_path)
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar

        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                palindrome = plot_pattern_analysis_radar('1234554321', backend='plotly')
                asymmetric = plot_pattern_analysis_radar('9753102468', backend='plotly')
            finally:
                reset_config()

        assert palindrome['metadata']['symmetry_score'] == 1.0
        assert asymmetric['metadata']['symmetry_score'] == 0.0

    def test_create_animation_writes_gif(self):
        from symergetics.visualization import create_animation

        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = create_animation(
                    [{'polyhedron': 'cuboctahedron'}] * 4,
                    output_filename='regression_rotation', fps=4)
            finally:
                reset_config()

            assert result['metadata']['type'] == 'animation'
            assert result['metadata']['frames'] == 4
            for file_path in result['files']:
                assert os.path.exists(file_path)
                assert os.path.getsize(file_path) > 0

    def test_remaining_plotly_renderers_covered(self):
        """Cover the rest of the new plotly backends."""
        from symergetics.visualization.mathematical import (
            plot_pattern_analysis,
            plot_ssrcd_analysis,
        )
        from symergetics.visualization.numbers import plot_mnemonic_visualization

        functions = [
            (plot_mnemonic_visualization, (1001,), {}),
            (plot_pattern_analysis, (1234554321, 'palindrome'), {}),
            (plot_ssrcd_analysis, (6,), {}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                results = [fn(*args, backend='plotly', **kwargs)
                           for fn, args, kwargs in functions]
            finally:
                reset_config()

            for result in results:
                assert result['metadata']['backend'] == 'plotly'
                for file_path in result['files']:
                    assert os.path.exists(file_path)
                    assert os.path.getsize(file_path) > 0

    def test_animation_input_validation(self):
        """create_animation rejects empty frame lists and unknown shapes."""
        from symergetics.visualization import create_animation

        with pytest.raises(ValueError):
            create_animation([], output_filename='empty')
        with pytest.raises(ValueError):
            create_animation([{'polyhedron': 'dodecahedron'}],
                             output_filename='bad_shape')
