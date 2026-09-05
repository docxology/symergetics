"""
Tests for core.constants module - Symergetics constants and values.

Tests mathematical constants, volume ratios, and cosmic scaling factors.
"""

import math

import pytest

from symergetics.core.constants import (
    COSMIC_ABUNDANCE,
    PHI,
    PI,
    SCHEHERAZADE_BASE,
    SQRT2,
    SQRT3,
    E,
    SymergeticsConstants,
)
from symergetics.core.numbers import SymergeticsNumber


class TestSymergeticsConstants:
    """Test SymergeticsConstants class."""

    def test_volume_ratios(self):
        """Test volume ratios for regular polyhedra."""
        # Tetrahedron should be 1 (unit volume)
        tetra_vol = SymergeticsConstants.get_volume_ratio('tetrahedron')
        assert tetra_vol.value == 1

        # Octahedron should be 4
        octa_vol = SymergeticsConstants.get_volume_ratio('octahedron')
        assert octa_vol.value == 4

        # Cube should be 3
        cube_vol = SymergeticsConstants.get_volume_ratio('cube')
        assert cube_vol.value == 3

        # Cuboctahedron should be 20
        cubocta_vol = SymergeticsConstants.get_volume_ratio('cuboctahedron')
        assert cubocta_vol.value == 20

        # Test invalid polyhedron
        with pytest.raises(ValueError):
            SymergeticsConstants.get_volume_ratio('invalid_polyhedron')

    def test_scheherazade_powers(self):
        """Test Scheherazade number calculations."""
        # Base case
        power1 = SymergeticsConstants.get_scheherazade_power(1)
        assert power1.value == 1001

        # Square
        power2 = SymergeticsConstants.get_scheherazade_power(2)
        assert power2.value == 1001 * 1001

        # Higher powers
        power3 = SymergeticsConstants.get_scheherazade_power(3)
        assert power3.value == 1001 ** 3

        # Test caching - should return same object
        power2_again = SymergeticsConstants.get_scheherazade_power(2)
        assert power2_again is power2

    def test_primorials(self):
        """Test primorial calculations."""
        # Small primorials
        primorial_2 = SymergeticsConstants.get_primorial(2)
        assert primorial_2.value == 2

        primorial_3 = SymergeticsConstants.get_primorial(3)
        assert primorial_3.value == 6  # 2 * 3

        primorial_5 = SymergeticsConstants.get_primorial(5)
        assert primorial_5.value == 30  # 2 * 3 * 5

        # Larger primorial
        primorial_13 = SymergeticsConstants.get_primorial(13)
        assert primorial_13.value == 30030  # 2 * 3 * 5 * 7 * 11 * 13

    def test_cosmic_scaling(self):
        """Test cosmic scaling relationships."""
        # Atomic diameters per inch
        atomic_scale = SymergeticsConstants.COSMIC_SCALING['atomic_diameters_per_inch']
        assert atomic_scale.value == 25_000_000_000

        # Earth circumference
        earth_degrees = SymergeticsConstants.COSMIC_SCALING['earth_circumference_degrees']
        assert earth_degrees.value == 360

        earth_seconds = SymergeticsConstants.COSMIC_SCALING['earth_circumference_seconds']
        assert earth_seconds.value == 1_296_000

    def test_cosmic_abundance_number(self):
        """Test Fuller's 14-illion cosmic abundance number."""
        abundance = SymergeticsConstants.COSMIC_ABUNDANCE_14_ILLION

        # Should be very large number (around 3.1 × 10^42)
        assert abundance.value > 10**40  # Correct magnitude

        # Test the factors function
        factors_info = SymergeticsConstants.cosmic_abundance_factors()

        assert factors_info['factor_count'] == 14
        assert len(factors_info['primes']) == 14
        assert len(factors_info['exponents']) == 14

        # Verify the number matches
        assert factors_info['cosmic_abundance'] == abundance

    def test_irrational_approximations(self):
        """Test irrational number approximations."""
        # π approximation
        pi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['pi']
        assert abs(float(pi_approx.value) - math.pi) < 1e-6

        # Golden ratio
        phi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['phi']
        golden_ratio = (1 + math.sqrt(5)) / 2
        assert abs(float(phi_approx.value) - golden_ratio) < 1e-6

        # Square root of 2
        sqrt2_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['sqrt2']
        assert abs(float(sqrt2_approx.value) - math.sqrt(2)) < 1e-6

        # Square root of 3
        sqrt3_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['sqrt3']
        assert abs(float(sqrt3_approx.value) - math.sqrt(3)) < 1e-6

    def test_diameter_to_edge_ratios(self):
        """Diameter/edge ratios: max vertex-pair distance in edge units."""
        tetra_ratio = SymergeticsConstants.DIAMETER_TO_EDGE_RATIOS['tetrahedron']
        assert tetra_ratio.value == 1

        # Octahedron diameter (opposite vertices) is √2 edges
        octa_ratio = SymergeticsConstants.DIAMETER_TO_EDGE_RATIOS['octahedron']
        assert abs(float(octa_ratio.value) - math.sqrt(2)) < 1e-6

        # Cube space diagonal is √3 edges
        cube_ratio = SymergeticsConstants.DIAMETER_TO_EDGE_RATIOS['cube']
        assert abs(float(cube_ratio.value) - math.sqrt(3)) < 1e-6

        # Cross-check against the actual vertex constants
        from symergetics.core.coordinates import OCTAHEDRON_VERTICES, urner_embedding
        points = [v.to_xyz(urner_embedding()) for v in OCTAHEDRON_VERTICES]
        dists = [sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
                 for i, a in enumerate(points) for b in points[i + 1:]]
        measured = max(dists) / min(dists)
        assert abs(measured - float(octa_ratio.value)) < 1e-6

    def test_vector_equilibrium_constants(self):
        """Test vector equilibrium (cuboctahedron) constants."""
        ve = SymergeticsConstants.VECTOR_EQUILIBRIUM

        assert ve['frequency_formula'].value == 12
        assert ve['surface_vectors'].value == 12
        assert ve['vertex_vectors'].value == 12
        assert ve['tetrahedral_cells'].value == 8
        assert ve['octahedral_cells'].value == 6

    def test_all_constants_method(self):
        """Test all_constants method returns proper structure."""
        all_consts = SymergeticsConstants.all_constants()

        # Should have all categories
        expected_categories = [
            'volume_ratios', 'scheherazade_powers', 'primorials',
            'cosmic_scaling', 'irrational_approximations'
        ]

        for category in expected_categories:
            assert category in all_consts
            assert isinstance(all_consts[category], dict)

    def test_by_category_method(self):
        """Test by_category method."""
        # Test volume ratios
        volumes = SymergeticsConstants.by_category('volume')
        assert 'tetrahedron' in volumes
        assert volumes['tetrahedron'] == 1

        # Test with different aliases
        volumes2 = SymergeticsConstants.by_category('volumes')
        assert volumes == volumes2

        # Test invalid category
        with pytest.raises(ValueError):
            SymergeticsConstants.by_category('invalid_category')


class TestConvenienceConstants:
    """Test convenience constant variables."""

    def test_phi_constant(self):
        """Test PHI convenience constant."""
        golden_ratio = (1 + math.sqrt(5)) / 2
        assert abs(float(PHI.value) - golden_ratio) < 1e-6

    def test_pi_constant(self):
        """Test PI convenience constant."""
        assert abs(float(PI.value) - math.pi) < 1e-6

    def test_e_constant(self):
        """Test E convenience constant."""
        assert abs(float(E.value) - math.e) < 1e-6

    def test_sqrt_constants(self):
        """Test square root convenience constants."""
        assert abs(float(SQRT2.value) - math.sqrt(2)) < 1e-6
        assert abs(float(SQRT3.value) - math.sqrt(3)) < 1e-6

    def test_scheherazade_base_constant(self):
        """Test SCHEHERAZADE_BASE convenience constant."""
        assert SCHEHERAZADE_BASE.value == 1001

    def test_cosmic_abundance_constant(self):
        """Test COSMIC_ABUNDANCE convenience constant."""
        # Should match the full calculation
        assert COSMIC_ABUNDANCE == SymergeticsConstants.COSMIC_ABUNDANCE_14_ILLION


    def test_prime_sieve_functionality(self):
        """Test the internal prime sieve functionality."""
        # Test _get_primes_up_to method
        primes = SymergeticsConstants._get_primes_up_to(10)
        assert primes == [2, 3, 5, 7]

        primes = SymergeticsConstants._get_primes_up_to(20)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19]

        # Test edge cases
        assert SymergeticsConstants._get_primes_up_to(1) == []
        assert SymergeticsConstants._get_primes_up_to(0) == []

    def test_cosmic_scale_factor_calculations(self):
        """Test cosmic scale factor calculations."""
        # Test available scaling factors
        factor = SymergeticsConstants.get_cosmic_scale_factor('inches', 'atomic_diameters')
        assert isinstance(factor, SymergeticsNumber)
        assert factor.value.numerator > 0

        # Test reverse conversion
        reverse_factor = SymergeticsConstants.get_cosmic_scale_factor('atomic_diameters', 'inches')
        assert isinstance(reverse_factor, SymergeticsNumber)
        # Reverse should be 1/factor
        assert abs(float(reverse_factor.value) - 1.0/float(factor.value)) < 1e-10

    def test_irrational_approximations_precision(self):
        """Test that irrational approximations have reasonable precision."""
        from math import e, pi, sqrt

        pi_approx = PI
        e_approx = E
        sqrt2_approx = SQRT2

        # Check that approximations are reasonably close
        assert abs(float(pi_approx.value) - pi) < 0.001
        assert abs(float(e_approx.value) - e) < 0.01
        assert abs(float(sqrt2_approx.value) - sqrt(2)) < 0.001

    def test_constants_caching(self):
        """Test that constants are properly cached."""
        # Get a value twice to test caching
        val1 = SymergeticsConstants.get_scheherazade_power(3)
        val2 = SymergeticsConstants.get_scheherazade_power(3)

        # Should be the same object (cached)
        assert val1 is val2

        # Test primorial caching
        prim1 = SymergeticsConstants.get_primorial(5)
        prim2 = SymergeticsConstants.get_primorial(5)

        assert prim1 is prim2

    def test_constants_by_category_comprehensive(self):
        """Test comprehensive category-based constant retrieval."""
        geometry_constants = SymergeticsConstants.by_category('geometry')
        assert isinstance(geometry_constants, dict)
        assert len(geometry_constants) > 0

        physics_constants = SymergeticsConstants.by_category('physics')
        assert isinstance(physics_constants, dict)

    def test_all_constants_comprehensive(self):
        """Test comprehensive all constants retrieval."""
        all_consts = SymergeticsConstants.all_constants()
        assert isinstance(all_consts, dict)

        # Count total constants across all categories
        total_constants = sum(len(category_constants) for category_constants in all_consts.values())
        assert total_constants > 10  # Should have many constants

        # Check that we have multiple categories
        assert len(all_consts) >= 3  # Should have at least 3 categories

    def test_vector_equilibrium_constants_detailed(self):
        """Test detailed vector equilibrium constants."""
        ve_constants = SymergeticsConstants.VECTOR_EQUILIBRIUM
        assert isinstance(ve_constants, dict)

        # Check specific VE constants
        expected_keys = ['frequency_formula', 'surface_vectors', 'edge_vectors', 'vertex_vectors', 'tetrahedral_cells', 'octahedral_cells']
        for key in expected_keys:
            assert key in ve_constants
            assert isinstance(ve_constants[key], SymergeticsNumber)

    def test_diameter_to_edge_ratios_detailed(self):
        """Test detailed diameter-to-edge ratios."""
        edge_ratios = SymergeticsConstants.DIAMETER_TO_EDGE_RATIOS
        assert isinstance(edge_ratios, dict)

        # Should contain ratios for different polyhedra
        assert len(edge_ratios) > 0
        for ratio in edge_ratios.values():
            assert isinstance(ratio, SymergeticsNumber)



class TestSymergeticsConstantsBasic:
    """Test basic SymergeticsConstants functionality."""

    def test_initialization(self):
        """Test constants initialization."""
        constants = SymergeticsConstants()
        assert constants is not None
        assert hasattr(SymergeticsConstants, 'VOLUME_RATIOS')
        assert hasattr(SymergeticsConstants, 'SCHEHERAZADE_POWERS')
        assert hasattr(SymergeticsConstants, 'PRIMORIALS')

    def test_volume_ratios(self):
        """Test volume ratio calculations."""
        ratios = SymergeticsConstants.VOLUME_RATIOS

        # Test that all ratios are positive
        for name, ratio in ratios.items():
            assert isinstance(ratio, SymergeticsNumber)
            assert ratio > 0, f"Volume ratio for {name} should be positive"

        # Test specific known ratios
        assert 'tetrahedron' in ratios
        assert 'octahedron' in ratios
        assert 'cube' in ratios
        assert 'cuboctahedron' in ratios

    def test_scheherazade_powers(self):
        """Test Scheherazade power calculations."""
        powers = SymergeticsConstants.SCHEHERAZADE_POWERS

        # Test that all powers are positive
        for power, value in powers.items():
            assert isinstance(value, SymergeticsNumber)
            assert value > 0, f"Scheherazade power {power} should be positive"

        # Test specific powers
        assert 1 in powers  # 1001^1 = 1001
        assert 2 in powers  # 1001^2
        assert 3 in powers  # 1001^3

    def test_primorials(self):
        """Test primorial calculations."""
        primorials = SymergeticsConstants.PRIMORIALS

        # Test that all primorials are positive
        for n, primorial in primorials.items():
            assert isinstance(primorial, SymergeticsNumber)
            assert primorial > 0, f"Primorial for n={n} should be positive"

        # Test specific primorials
        assert 2 in primorials  # 2# = 2
        assert 3 in primorials  # 3# = 6
        assert 5 in primorials  # 5# = 30
        assert 7 in primorials  # 7# = 210
        assert 11 in primorials  # 11# = 2310


class TestSymergeticsConstantsMathematical:
    """Test mathematical properties of constants."""

    def test_volume_ratio_relationships(self):
        """Test relationships between volume ratios."""
        ratios = SymergeticsConstants.VOLUME_RATIOS

        # Tetrahedron should be the base unit (volume = 1)
        tetra_vol = ratios['tetrahedron']
        assert tetra_vol == 1

        # Octahedron should be 4 times tetrahedron
        octa_vol = ratios['octahedron']
        assert octa_vol == 4

        # Cube should be 3 times tetrahedron
        cube_vol = ratios['cube']
        assert cube_vol == 3

    def test_scheherazade_power_consistency(self):
        """Test consistency of Scheherazade powers."""
        powers = SymergeticsConstants.SCHEHERAZADE_POWERS

        # Test that powers are positive and increasing
        power_values = list(powers.values())
        for i in range(1, len(power_values)):
            assert power_values[i] > power_values[i-1], f"Power {i+1} should be greater than power {i}"

        # Test that all values are SymergeticsNumbers
        for power, value in powers.items():
            assert isinstance(value, SymergeticsNumber)
            assert value > 0

    def test_primorial_consistency(self):
        """Test consistency of primorial calculations."""
        primorials = SymergeticsConstants.PRIMORIALS

        # Test that primorials follow the correct pattern
        expected_primorials = {
            2: 2,
            3: 6,
            5: 30,
            7: 210,
            11: 2310
        }

        for n, expected in expected_primorials.items():
            if n in primorials:
                assert primorials[n] == expected, f"Primorial {n}# should equal {expected}"


class TestSymergeticsConstantsIrrational:
    """Test irrational number approximations."""

    def test_pi_approximation(self):
        """Test pi approximation."""
        pi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['pi']

        assert isinstance(pi_approx, SymergeticsNumber)
        assert abs(pi_approx.to_float() - math.pi) < 1e-10

    def test_phi_approximation(self):
        """Test golden ratio approximation."""
        phi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['phi']

        assert isinstance(phi_approx, SymergeticsNumber)
        golden_ratio = (1 + math.sqrt(5)) / 2
        assert abs(phi_approx.to_float() - golden_ratio) < 1e-10

    def test_e_approximation(self):
        """Test Euler's number approximation."""
        e_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['e']

        assert isinstance(e_approx, SymergeticsNumber)
        assert abs(e_approx.to_float() - math.e) < 1e-10

    def test_sqrt2_approximation(self):
        """Test square root of 2 approximation."""
        sqrt2_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['sqrt2']

        assert isinstance(sqrt2_approx, SymergeticsNumber)
        assert abs(sqrt2_approx.to_float() - math.sqrt(2)) < 1e-10


class TestSymergeticsConstantsMethods:
    """Test methods of SymergeticsConstants."""

    def test_all_constants_method(self):
        """Test all_constants method."""
        all_constants = SymergeticsConstants.all_constants()

        assert isinstance(all_constants, dict)
        assert 'volume_ratios' in all_constants
        assert 'scheherazade_powers' in all_constants
        assert 'primorials' in all_constants
        assert 'irrational_approximations' in all_constants

    def test_by_category_method(self):
        """Test by_category method."""
        # Test getting volume ratios
        volume_ratios = SymergeticsConstants.by_category('volume_ratios')
        assert isinstance(volume_ratios, dict)
        assert 'tetrahedron' in volume_ratios

        # Test getting primorials
        primorials = SymergeticsConstants.by_category('primorials')
        assert isinstance(primorials, dict)
        assert 2 in primorials

    def test_get_volume_ratio(self):
        """Test get_volume_ratio method."""
        tetra_vol = SymergeticsConstants.get_volume_ratio('tetrahedron')
        assert tetra_vol == 1

        octa_vol = SymergeticsConstants.get_volume_ratio('octahedron')
        assert octa_vol == 4

    def test_get_scheherazade_power(self):
        """Test get_scheherazade_power method."""
        power_1 = SymergeticsConstants.get_scheherazade_power(1)
        assert power_1 == 1001

        power_2 = SymergeticsConstants.get_scheherazade_power(2)
        assert power_2 == 1002001

    def test_get_primorial(self):
        """Test get_primorial method."""
        prim_2 = SymergeticsConstants.get_primorial(2)
        assert prim_2 == 2

        prim_3 = SymergeticsConstants.get_primorial(3)
        assert prim_3 == 6

        prim_5 = SymergeticsConstants.get_primorial(5)
        assert prim_5 == 30


class TestSymergeticsConstantsEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_category(self):
        """Test by_category with invalid category."""
        with pytest.raises(ValueError):
            SymergeticsConstants.by_category('invalid_category')

    def test_invalid_polyhedron(self):
        """Test get_volume_ratio with invalid polyhedron."""
        with pytest.raises(ValueError):
            SymergeticsConstants.get_volume_ratio('invalid_polyhedron')


class TestSymergeticsConstantsIntegration:
    """Test integration with other components."""

    def test_with_symergetics_numbers(self):
        """Test interaction with SymergeticsNumber."""
        # Test that constants work with SymergeticsNumber arithmetic
        tetra_vol = SymergeticsConstants.VOLUME_RATIOS['tetrahedron']
        octa_vol = SymergeticsConstants.VOLUME_RATIOS['octahedron']

        # Test arithmetic operations
        sum_vol = tetra_vol + octa_vol
        assert isinstance(sum_vol, SymergeticsNumber)
        assert sum_vol == 5

        # Test multiplication
        double_tetra = tetra_vol * 2
        assert double_tetra == 2

    def test_precision_consistency(self):
        """Test that precision is consistent across calculations."""
        # Test that irrational approximations maintain precision
        pi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['pi']
        phi_approx = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['phi']

        # Both should have high precision
        assert len(str(pi_approx.numerator)) > 5
        assert len(str(phi_approx.numerator)) > 5

if __name__ == "__main__":
    pytest.main([__file__])
