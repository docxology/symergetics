"""
Comprehensive tests for SymergeticsConstants and related functionality.
"""

import pytest
import math
from symergetics.core.constants import SymergeticsConstants
from symergetics.core.numbers import SymergeticsNumber


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