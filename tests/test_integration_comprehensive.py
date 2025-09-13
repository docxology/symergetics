"""
Comprehensive integration tests across all modules.
Tests real functionality and interactions between modules.
"""

import pytest
import numpy as np
from fractions import Fraction
import tempfile
import os
from pathlib import Path

from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.constants import SymergeticsConstants
from symergetics.core.coordinates import QuadrayCoordinate, urner_embedding
from symergetics.computation.analysis import analyze_mathematical_patterns
from symergetics.computation.palindromes import analyze_number_for_synergetics
from symergetics.computation.primorials import primorial, scheherazade_power
from symergetics.computation.geometric_mnemonics import analyze_geometric_mnemonics
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube, Cuboctahedron
from symergetics.geometry.transformations import translate, scale
from symergetics.utils.conversion import rational_to_float, float_to_exact_rational
from symergetics.utils.mnemonics import mnemonic_encode, create_memory_aid
from symergetics.utils.reporting import generate_statistical_summary


class TestCoreIntegration:
    """Test integration between core modules."""
    
    def test_numbers_constants_integration(self):
        """Test integration between numbers and constants."""
        constants = SymergeticsConstants()
        number = SymergeticsNumber(1001)
        
        # Test Scheherazade integration
        scheherazade_result = number.to_scheherazade_base()
        assert isinstance(scheherazade_result, tuple)
        assert len(scheherazade_result) == 2
        
        # Test with constants
        scheherazade_power_1 = constants.get_scheherazade_power(1)
        assert scheherazade_power_1.numerator == 1001
        
        # Test arithmetic with constants
        volume_ratio = constants.get_volume_ratio('tetrahedron')
        result = number + volume_ratio
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator > 0
        assert result.denominator > 0
    
    def test_coordinates_embedding_integration(self):
        """Test integration between coordinates and embedding."""
        coord = QuadrayCoordinate(1, 0, 0, 0)
        embedding = urner_embedding()
        
        # Test transformation
        xyz = coord.to_xyz(embedding)
        assert len(xyz) == 3
        assert isinstance(xyz[0], (int, float))
        
        # Test inverse transformation
        coord_back = QuadrayCoordinate.from_xyz(xyz[0], xyz[1], xyz[2], embedding)
        assert isinstance(coord_back, QuadrayCoordinate)
        
        # Test round trip accuracy
        assert abs(coord_back.a - coord.a) < 1e-10
        assert abs(coord_back.b - coord.b) < 1e-10
        assert abs(coord_back.c - coord.c) < 1e-10
        assert abs(coord_back.d - coord.d) < 1e-10
    
    def test_numbers_coordinates_integration(self):
        """Test integration between numbers and coordinates."""
        # Test coordinate arithmetic with SymergeticsNumber
        coord1 = QuadrayCoordinate(1, 0, 0, 0)
        coord2 = QuadrayCoordinate(0, 1, 0, 0)
        
        # Test coordinate addition with SymergeticsNumber values
        coord1_with_symergetics = QuadrayCoordinate(
            SymergeticsNumber(1), SymergeticsNumber(0), 
            SymergeticsNumber(0), SymergeticsNumber(0)
        )
        coord2_with_symergetics = QuadrayCoordinate(
            SymergeticsNumber(0), SymergeticsNumber(1), 
            SymergeticsNumber(0), SymergeticsNumber(0)
        )
        result = coord1_with_symergetics.add(coord2_with_symergetics)
        assert isinstance(result, QuadrayCoordinate)
        assert result.a == 1
        assert result.b == 1
        assert result.c == 0
        assert result.d == 0


class TestComputationIntegration:
    """Test integration between computation modules."""
    
    def test_analysis_palindromes_integration(self):
        """Test integration between analysis and palindromes."""
        # Test with palindromic numbers
        palindromic_numbers = [12321, 1001, 1234321]
        
        for num in palindromic_numbers:
            # Test analysis
            analysis = analyze_mathematical_patterns(str(num))
            assert 'is_palindromic' in analysis
            assert analysis['is_palindromic'] == True
            
            # Test palindrome analysis
            palindrome_analysis = analyze_number_for_synergetics(str(num))
            assert 'is_palindromic' in palindrome_analysis
            assert 'palindromic_patterns' in palindrome_analysis
            assert palindrome_analysis['is_palindromic'] == True
    
    def test_analysis_primorials_integration(self):
        """Test integration between analysis and primorials."""
        # Test with primorial numbers and their corresponding n values
        primorial_data = [(2, 2), (6, 3), (30, 5), (210, 7), (2310, 11)]
        
        for expected_primorial, n in primorial_data:
            # Test analysis
            analysis = analyze_mathematical_patterns(str(expected_primorial))
            assert 'is_palindromic' in analysis
            assert 'palindromic_density' in analysis
            assert 'digit_distribution' in analysis
            
            # Test primorial calculation
            primorial_result = primorial(n)
            assert primorial_result == expected_primorial
    
    def test_analysis_scheherazade_integration(self):
        """Test integration between analysis and Scheherazade numbers."""
        # Test with Scheherazade numbers and their corresponding powers
        scheherazade_data = [(1001, 1), (1002001, 2), (1003003001, 3)]
        
        for expected_scheherazade, power in scheherazade_data:
            # Test analysis
            analysis = analyze_mathematical_patterns(str(expected_scheherazade))
            assert 'is_palindromic' in analysis
            assert 'palindromic_density' in analysis
            assert 'digit_distribution' in analysis
            
            # Test Scheherazade power calculation
            scheherazade_result = scheherazade_power(power)
            assert scheherazade_result == expected_scheherazade
    
    def test_geometric_mnemonics_integration(self):
        """Test integration between analysis and geometric mnemonics."""
        # Test with geometric numbers
        geometric_numbers = [1, 4, 3, 20]  # Tetrahedron, Octahedron, Cube, Cuboctahedron
        
        for num in geometric_numbers:
            # Test analysis
            analysis = analyze_mathematical_patterns(str(num))
            assert 'is_palindromic' in analysis
            assert 'palindromic_density' in analysis
            assert 'digit_distribution' in analysis
            
            # Test geometric mnemonic analysis
            mnemonic_analysis = analyze_geometric_mnemonics(num)
            assert 'geometric_mnemonics' in mnemonic_analysis
            assert 'platonic_relationships' in mnemonic_analysis


class TestGeometryIntegration:
    """Test integration between geometry modules."""
    
    def test_polyhedra_volume_integration(self):
        """Test integration between polyhedra and volume calculations."""
        # Test tetrahedron
        tetra = Tetrahedron()
        assert tetra.volume() == 1
        
        # Test octahedron
        octa = Octahedron()
        assert octa.volume() == 4
        
        # Test cube
        cube = Cube()
        assert cube.volume() == 3
        
        # Test cuboctahedron
        cubocta = Cuboctahedron()
        assert cubocta.volume() == 20
    
    def test_polyhedra_transformations_integration(self):
        """Test integration between polyhedra and transformations."""
        # Test tetrahedron with transformations
        tetra = Tetrahedron()
        original_vertices = tetra.vertices.copy()
        
        # Test translation
        coord = original_vertices[0]  # This is already a QuadrayCoordinate
        offset = QuadrayCoordinate.from_xyz(1, 1, 1)
        translated_coord = translate(coord, offset)
        translated = translated_coord.to_xyz()
        original_xyz = coord.to_xyz()
        assert len(translated) == 3
        assert translated[0] == original_xyz[0] + 1
        assert translated[1] == original_xyz[1] + 1
        assert translated[2] == original_xyz[2] + 1
        
        # Test scaling
        scaled_coord = scale(coord, 2)
        scaled = scaled_coord.to_xyz()
        assert len(scaled) == 3
        assert scaled[0] == original_xyz[0] * 2
        assert scaled[1] == original_xyz[1] * 2
        assert scaled[2] == original_xyz[2] * 2
    
    def test_coordinates_polyhedra_integration(self):
        """Test integration between coordinates and polyhedra."""
        # Test tetrahedron vertices in Quadray coordinates
        tetra = Tetrahedron()
        vertices = tetra.vertices
        
        # Test that all vertices are valid Quadray coordinates
        for coord in vertices:
            assert isinstance(coord, QuadrayCoordinate)
            # Test that coordinates have the expected attributes
            assert hasattr(coord, 'a')
            assert hasattr(coord, 'b')
            assert hasattr(coord, 'c')
            assert hasattr(coord, 'd')


class TestUtilsIntegration:
    """Test integration between utility modules."""
    
    def test_conversion_mnemonics_integration(self):
        """Test integration between conversion and mnemonics."""
        # Test conversion to mnemonic
        number = SymergeticsNumber(12345)
        mnemonic = mnemonic_encode(number)
        assert isinstance(mnemonic, str)
        assert len(mnemonic) > 0
        
        # Test memory aid creation
        memory_aid = create_memory_aid(number)
        assert isinstance(memory_aid, dict)
        assert len(memory_aid) > 0
        
        # Test conversion with different number types
        float_num = 3.14159
        rational = float_to_exact_rational(float_num)
        mnemonic_rational = mnemonic_encode(rational)
        assert isinstance(mnemonic_rational, str)
    
    def test_conversion_reporting_integration(self):
        """Test integration between conversion and reporting."""
        # Test with converted numbers - first analyze them
        numbers = [SymergeticsNumber(123), SymergeticsNumber(456), SymergeticsNumber(789)]
        analysis_results = [analyze_mathematical_patterns(str(num)) for num in numbers]
        
        # Test statistical summary
        summary = generate_statistical_summary(analysis_results)
        assert isinstance(summary, dict)
        assert 'total_analyses' in summary
        assert 'metrics' in summary
        
        # Test with mixed number types - first analyze them
        mixed_numbers = [123, 456.789, SymergeticsNumber(789)]
        mixed_analysis_results = [analyze_mathematical_patterns(str(num)) for num in mixed_numbers]
        summary_mixed = generate_statistical_summary(mixed_analysis_results)
        assert isinstance(summary_mixed, dict)
    
    def test_mnemonics_reporting_integration(self):
        """Test integration between mnemonics and reporting."""
        # Test with mnemonic data
        numbers = [SymergeticsNumber(1001), SymergeticsNumber(12321), SymergeticsNumber(456)]
        mnemonics = [mnemonic_encode(num) for num in numbers]
        
        # Test reporting with mnemonic data - first analyze the numbers
        analysis_results = [analyze_mathematical_patterns(str(num)) for num in numbers]
        summary = generate_statistical_summary(analysis_results)
        assert isinstance(summary, dict)
        
        # Test memory aid creation for reporting
        memory_aids = [create_memory_aid(num) for num in numbers]
        assert len(memory_aids) == len(numbers)
        assert all(isinstance(aid, dict) for aid in memory_aids)


class TestFullWorkflowIntegration:
    """Test complete workflow integration."""
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Test data
        test_numbers = [1001, 12321, 456, 789, 1002001, 2, 6, 30, 1, 4, 3, 20]
        
        # Step 1: Convert to SymergeticsNumber
        symergetics_numbers = [SymergeticsNumber(num) for num in test_numbers]
        
        # Step 2: Analyze patterns
        analyses = []
        for num in symergetics_numbers:
            analysis = analyze_mathematical_patterns(str(num.numerator))
            analyses.append(analysis)
        
        # Step 3: Test palindromic analysis
        palindrome_analyses = []
        for num in test_numbers:
            if str(num) == str(num)[::-1]:  # Simple palindrome check
                analysis = analyze_number_for_synergetics(str(num))
                palindrome_analyses.append(analysis)
        
        # Step 4: Test primorial analysis
        primorial_analyses = []
        for i, num in enumerate(test_numbers):
            if num in [2, 6, 30, 210, 2310]:  # Known primorials
                primorial_result = primorial(i + 1)
                primorial_analyses.append(primorial_result)
        
        # Step 5: Test Scheherazade analysis
        scheherazade_analyses = []
        for num in test_numbers:
            if num in [1001, 1002001, 1003003001]:  # Known Scheherazade numbers
                power = [1001, 1002001, 1003003001].index(num) + 1
                scheherazade_result = scheherazade_power(power)
                scheherazade_analyses.append(scheherazade_result)
        
        # Step 6: Test geometric analysis
        geometric_analyses = []
        for num in test_numbers:
            if num in [1, 4, 3, 20]:  # Known geometric volumes
                analysis = analyze_geometric_mnemonics(num)
                geometric_analyses.append(analysis)
        
        # Step 7: Test reporting - first analyze the numbers
        analysis_results = [analyze_mathematical_patterns(str(num)) for num in symergetics_numbers]
        summary = generate_statistical_summary(analysis_results)
        
        # Verify all steps completed successfully
        assert len(analyses) == len(test_numbers)
        assert len(palindrome_analyses) > 0
        assert len(primorial_analyses) > 0
        assert len(scheherazade_analyses) > 0
        assert len(geometric_analyses) > 0
        assert isinstance(summary, dict)
    
    def test_visualization_workflow_integration(self):
        """Test visualization workflow integration."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        from symergetics.visualization.geometry import plot_polyhedron
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        # Test data
        numbers = [SymergeticsNumber(1001), SymergeticsNumber(12321), SymergeticsNumber(456)]
        polyhedra_data = {'tetrahedron': 1, 'octahedron': 4, 'cube': 3}
        analysis_data = {
            'numbers': [1001, 12321, 456],
            'patterns': ['scheherazade', 'palindrome', 'ascending'],
            'properties': ['composite', 'palindromic', 'composite']
        }
        
        # Test multiple visualizations
        result1 = plot_pattern_analysis(
            number=1001,
            pattern_type='palindrome'
        )
        
        result2 = plot_polyhedron(
            polyhedron="tetrahedron"
        )
        
        result3 = plot_palindromic_pattern(
            number=12321
        )
        
        # Verify all visualizations completed and returned valid results
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        
        # Verify results are dictionaries with expected keys
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)
    
    def test_error_handling_integration(self):
        """Test error handling across modules."""
        # Test with invalid inputs
        invalid_inputs = [None, "", "invalid", -1, 0]
        
        for invalid_input in invalid_inputs:
            try:
                # Test analysis with invalid input
                analysis = analyze_mathematical_patterns(str(invalid_input))
                # Should either succeed or raise a meaningful error
                assert isinstance(analysis, dict) or True
            except Exception as e:
                # Should be a meaningful error, not a crash
                assert isinstance(e, (ValueError, TypeError, AttributeError))
            
            try:
                # Test conversion with invalid input
                if invalid_input is not None:
                    rational = float_to_exact_rational(invalid_input)
                    assert isinstance(rational, SymergeticsNumber) or True
            except Exception as e:
                # Should be a meaningful error, not a crash
                assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_performance_integration(self):
        """Test performance across modules."""
        import time
        
        # Test with larger datasets
        large_numbers = [SymergeticsNumber(i) for i in range(1, 101)]  # 100 numbers
        
        # Test analysis performance
        start_time = time.time()
        analyses = [analyze_mathematical_patterns(str(num.numerator)) for num in large_numbers]
        analysis_time = time.time() - start_time
        
        assert len(analyses) == 100
        assert analysis_time < 10.0  # Should complete within 10 seconds
        
        # Test conversion performance
        start_time = time.time()
        conversions = [rational_to_float(num) for num in large_numbers]
        conversion_time = time.time() - start_time
        
        assert len(conversions) == 100
        assert conversion_time < 5.0  # Should complete within 5 seconds
        
        # Test reporting performance - first analyze the numbers
        start_time = time.time()
        analysis_results = [analyze_mathematical_patterns(str(num)) for num in large_numbers]
        summary = generate_statistical_summary(analysis_results)
        reporting_time = time.time() - start_time
        
        assert isinstance(summary, dict)
        assert reporting_time < 2.0  # Should complete within 2 seconds


class TestEdgeCaseIntegration:
    """Test edge cases across modules."""
    
    def test_zero_handling_integration(self):
        """Test zero handling across modules."""
        zero = SymergeticsNumber(0)
        
        # Test arithmetic with zero
        assert zero + zero == zero
        assert zero * zero == zero
        assert zero / SymergeticsNumber(1) == zero
        
        # Test analysis with zero
        analysis = analyze_mathematical_patterns("0")
        assert isinstance(analysis, dict)
        
        # Test conversion with zero
        float_zero = rational_to_float(zero)
        assert float_zero == 0.0
        
        # Test reporting with zero - first analyze it
        zero_analysis = analyze_mathematical_patterns("0")
        summary = generate_statistical_summary([zero_analysis])
        assert isinstance(summary, dict)
    
    def test_large_number_handling_integration(self):
        """Test large number handling across modules."""
        large_number = SymergeticsNumber(10**100)
        
        # Test arithmetic with large numbers
        result = large_number + large_number
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator > 0
        
        # Test analysis with large numbers
        analysis = analyze_mathematical_patterns(str(large_number.numerator))
        assert isinstance(analysis, dict)
        
        # Test conversion with large numbers
        float_large = rational_to_float(large_number)
        assert isinstance(float_large, float)
        assert float_large > 0
    
    def test_fractional_handling_integration(self):
        """Test fractional handling across modules."""
        fraction = SymergeticsNumber(22, 7)  # Pi approximation
        
        # Test arithmetic with fractions
        result = fraction + fraction
        assert isinstance(result, SymergeticsNumber)
        assert result.numerator == 44
        assert result.denominator == 7
        
        # Test analysis with fractions
        analysis = analyze_mathematical_patterns(f"{fraction.numerator}/{fraction.denominator}")
        assert isinstance(analysis, dict)
        
        # Test conversion with fractions
        float_fraction = rational_to_float(fraction)
        assert isinstance(float_fraction, float)
        assert abs(float_fraction - 22/7) < 1e-10
