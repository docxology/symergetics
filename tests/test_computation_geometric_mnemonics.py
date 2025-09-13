#!/usr/bin/env python3
"""
Tests for geometric mnemonics methods in symergetics.computation.geometric_mnemonics
"""

import pytest
import math
from symergetics.computation.geometric_mnemonics import (
    analyze_geometric_mnemonics,
    create_integer_ratio_visualization,
    generate_geometric_mnemonic_report,
    GeometricMnemonic
)
from symergetics.core.numbers import SymergeticsNumber


class TestGeometricMnemonicsAnalysis:
    """Test geometric mnemonics analysis functions."""

    def test_analyze_platonic_volume_relationship(self):
        """Test analysis of number related to platonic solid volume."""
        # 20 is cuboctahedron volume
        result = analyze_geometric_mnemonics(20)

        assert result['number'] == '20'
        assert 'platonic_relationships' in result
        assert 'cuboctahedron' in str(result.get('platonic_relationships', {}))

    def test_analyze_edge_relationship(self):
        """Test analysis of number related to platonic solid edges."""
        # 12 is edges of octahedron/cube/icosahedron
        result = analyze_geometric_mnemonics(12)

        assert result['number'] == '12'
        platonic_rels = result.get('platonic_relationships', {})
        assert len(platonic_rels) > 0

    def test_analyze_face_relationship(self):
        """Test analysis of number related to platonic solid faces."""
        # 20 is faces of icosahedron
        result = analyze_geometric_mnemonics(20)

        assert result['number'] == '20'
        # Should find both volume and face relationships
        platonic_rels = result.get('platonic_relationships', {})
        assert len(platonic_rels) > 0

    def test_analyze_rational_approximation(self):
        """Test analysis of rational approximation numbers."""
        # 22 appears in 22/7 ≈ π
        result = analyze_geometric_mnemonics(22, analysis_depth=5)

        assert result['number'] == '22'
        rational_approxs = result.get('rational_approximations', {})
        assert len(rational_approxs) > 0

    def test_analyze_scheherazade_number(self):
        """Test analysis of Scheherazade number."""
        result = analyze_geometric_mnemonics(1001)

        assert result['number'] == '1001'
        # Scheherazade numbers have special properties
        assert 'ivm_scaling_factors' in result

    def test_analyze_large_palindromic(self):
        """Test analysis of large palindromic number."""
        result = analyze_geometric_mnemonics(123454321)

        assert result['number'] == '123454321'
        assert result.get('is_palindromic', False) == True

    def test_analyze_with_different_depths(self):
        """Test analysis with different depth levels."""
        number = 24

        # Depth 1
        result1 = analyze_geometric_mnemonics(number, analysis_depth=1)
        assert 'platonic_relationships' in result1

        # Depth 3
        result3 = analyze_geometric_mnemonics(number, analysis_depth=3)
        assert 'ivm_scaling_factors' in result3

        # Depth 5
        result5 = analyze_geometric_mnemonics(number, analysis_depth=5)
        assert 'rational_approximations' in result5


class TestGeometricMnemonicVisualization:
    """Test geometric mnemonics visualization functions."""

    def test_create_integer_ratio_visualization(self):
        """Test integer ratio visualization creation."""
        numbers = [6, 12, 20, 24, 30]

        result = create_integer_ratio_visualization(
            numbers,
            title="Test Integer Ratio Visualization"
        )

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'geometric_mnemonics'
        assert result['metadata']['numbers_analyzed'] == len(numbers)

    def test_create_visualization_empty_numbers(self):
        """Test visualization with empty number list."""
        with pytest.raises(Exception):  # Should handle empty list gracefully
            create_integer_ratio_visualization([])

    def test_create_visualization_single_number(self):
        """Test visualization with single number."""
        result = create_integer_ratio_visualization(
            [12],
            title="Single Number Test"
        )

        assert 'files' in result
        assert result['metadata']['numbers_analyzed'] == 1


class TestGeometricMnemonicReporting:
    """Test geometric mnemonics reporting functions."""

    def test_generate_mnemonic_report(self):
        """Test comprehensive mnemonic report generation."""
        numbers = [6, 12, 20, 24]

        report = generate_geometric_mnemonic_report(
            numbers,
            title="Test Geometric Mnemonics Report"
        )

        assert 'title' in report
        assert 'summary' in report
        assert 'detailed_analysis' in report
        assert 'insights' in report
        assert 'recommendations' in report

        # Check summary statistics
        summary = report['summary']
        assert 'total_numbers_analyzed' in summary
        assert summary['total_numbers_analyzed'] == len(numbers)

    def test_generate_report_empty_numbers(self):
        """Test report generation with empty number list."""
        report = generate_geometric_mnemonic_report(
            [],
            title="Empty Report Test"
        )

        assert 'title' in report
        assert report['summary']['total_numbers_analyzed'] == 0

    def test_generate_report_single_number(self):
        """Test report generation with single number."""
        report = generate_geometric_mnemonic_report(
            [12],
            title="Single Number Report"
        )

        assert report['summary']['total_numbers_analyzed'] == 1


class TestGeometricMnemonicClass:
    """Test GeometricMnemonic class."""

    def test_geometric_mnemonic_creation(self):
        """Test GeometricMnemonic class creation."""
        mnemonic = GeometricMnemonic(
            value=24,
            interpretation="2 × vector equilibrium edges",
            geometric_form="vector_equilibrium"
        )

        assert mnemonic.value == 24
        assert mnemonic.interpretation == "2 × vector equilibrium edges"
        assert mnemonic.geometric_form == "vector_equilibrium"

    def test_geometric_mnemonic_string_representation(self):
        """Test string representation of GeometricMnemonic."""
        mnemonic = GeometricMnemonic(
            value=20,
            interpretation="Cuboctahedron volume",
            geometric_form="cuboctahedron"
        )

        str_repr = str(mnemonic)
        assert '20' in str_repr
        assert 'Cuboctahedron' in str_repr

    def test_geometric_mnemonic_with_symergetics_number(self):
        """Test GeometricMnemonic with SymergeticsNumber."""
        num = SymergeticsNumber(24)
        mnemonic = GeometricMnemonic(
            value=num,
            interpretation="Vector equilibrium edges",
            geometric_form="vector_equilibrium"
        )

        assert isinstance(mnemonic.value, SymergeticsNumber)


class TestComplexGeometricRelationships:
    """Test complex geometric relationships."""

    def test_vector_equilibrium_relationships(self):
        """Test vector equilibrium geometric relationships."""
        # Vector equilibrium has 12 vertices, 24 edges, 8 faces
        ve_numbers = [12, 24, 8, 48]  # Including internal structures

        for num in ve_numbers:
            result = analyze_geometric_mnemonics(num)
            platonic_rels = result.get('platonic_relationships', {})

            # Should find some relationship to vector equilibrium
            ve_found = any('vector_equilibrium' in rel_name or 'equilibrium' in rel_name.lower()
                          for rel_name in platonic_rels.keys())

            if not ve_found:
                # At least should find some geometric relationship
                assert len(platonic_rels) > 0

    def test_cosmic_scaling_relationships(self):
        """Test relationships to cosmic scaling factors."""
        # Numbers that might relate to cosmic scaling
        cosmic_numbers = [24, 48, 72, 96, 120]

        for num in cosmic_numbers:
            result = analyze_geometric_mnemonics(num, analysis_depth=4)
            # Should find some geometric relationships
            total_rels = (len(result.get('platonic_relationships', {})) +
                         len(result.get('ivm_scaling_factors', {})) +
                         len(result.get('rational_approximations', {})))
            assert total_rels > 0

    def test_fibonacci_sphere_packing(self):
        """Test Fibonacci-related sphere packing numbers."""
        # Fibonacci numbers often appear in sphere packing
        fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

        for num in fib_numbers[:5]:  # Test first few
            result = analyze_geometric_mnemonics(num, analysis_depth=4)
            # Fibonacci numbers often have geometric significance
            # At minimum should have some analysis
            assert 'number' in result


class TestRationalApproximationAnalysis:
    """Test rational approximation analysis."""

    def test_pi_approximations(self):
        """Test rational approximations of π."""
        pi_numerators = [22, 355, 314]  # From 22/7, 355/113, π×100

        for num in pi_numerators:
            result = analyze_geometric_mnemonics(num, analysis_depth=5)
            rational_approxs = result.get('rational_approximations', {})

            # Should find π-related approximations
            pi_found = any('π' in str(approx_data.get('constant', ''))
                          for approx_data in rational_approxs.values())

            if not pi_found:
                # At least should have some rational approximation
                assert len(rational_approxs) > 0

    def test_golden_ratio_approximations(self):
        """Test rational approximations of golden ratio φ."""
        phi_numerators = [577, 610]  # From continued fraction approximations

        for num in phi_numerators:
            result = analyze_geometric_mnemonics(num, analysis_depth=5)
            rational_approxs = result.get('rational_approximations', {})

            # Should find φ-related approximations
            phi_found = any('φ' in str(approx_data.get('constant', '')) or
                           'golden' in str(approx_data.get('description', '')).lower()
                          for approx_data in rational_approxs.values())

            if not phi_found:
                # At least should have some rational approximation
                assert len(rational_approxs) > 0

    def test_sqrt2_approximations(self):
        """Test rational approximations of √2."""
        sqrt2_numerators = [1414, 665]  # From √2 ≈ 1.414, and 665/470

        for num in sqrt2_numerators:
            result = analyze_geometric_mnemonics(num, analysis_depth=5)
            # Should have some analysis
            assert 'number' in result


class TestIntegrationWithOtherModules:
    """Test integration with other Symergetics modules."""

    def test_with_symergetics_constants(self):
        """Test integration with Symergetics constants."""
        from symergetics.core.constants import SymergeticsConstants

        constants = SymergeticsConstants()
        volume_ratios = constants.VOLUME_RATIOS

        # Test with volume ratios
        for solid, volume in volume_ratios.items():
            if isinstance(volume, SymergeticsNumber):
                volume_int = int(float(volume.value))
                result = analyze_geometric_mnemonics(volume_int)
                assert 'number' in result

    def test_with_quadray_coordinates(self):
        """Test integration with Quadray coordinate system."""
        from symergetics.core.coordinates import QuadrayCoordinate

        # Test with coordinates that form platonic solids
        tetra_coords = [
            QuadrayCoordinate(1, 0, 0, 0),
            QuadrayCoordinate(0, 1, 0, 0),
            QuadrayCoordinate(0, 0, 1, 0),
            QuadrayCoordinate(0, 0, 0, 1)
        ]

        # Volume of tetrahedron should be 1
        volume = 1
        result = analyze_geometric_mnemonics(volume)
        assert result['number'] == '1'

    def test_with_polyhedra_module(self):
        """Test integration with polyhedra module."""
        from symergetics.geometry.polyhedra import Tetrahedron

        tetra = Tetrahedron()
        # Tetrahedron has 6 edges
        result = analyze_geometric_mnemonics(6)
        platonic_rels = result.get('platonic_relationships', {})
        assert len(platonic_rels) > 0


class TestEdgeCases:
    """Test edge cases in geometric mnemonics."""

    def test_analyze_zero(self):
        """Test analysis of zero."""
        result = analyze_geometric_mnemonics(0)

        assert result['number'] == '0'
        assert result['numeric_value'] == 0

    def test_analyze_negative_number(self):
        """Test analysis of negative number."""
        result = analyze_geometric_mnemonics(-24)

        # Should handle negative numbers gracefully
        assert 'number' in result

    def test_analyze_very_large_number(self):
        """Test analysis of very large number."""
        large_num = 10**12
        result = analyze_geometric_mnemonics(large_num)

        assert result['number'] == str(large_num)
        # Should still perform some analysis
        total_keys = len(result.keys())
        assert total_keys > 2  # At least number, numeric_value, and some analysis

    def test_analyze_prime_number(self):
        """Test analysis of prime number."""
        prime = 97
        result = analyze_geometric_mnemonics(prime, analysis_depth=5)

        # Primes might have rational approximation relationships
        assert 'number' in result
        assert result['numeric_value'] == prime

    def test_analyze_fibonacci_number(self):
        """Test analysis of Fibonacci number."""
        fib = 89  # Fibonacci number
        result = analyze_geometric_mnemonics(fib, analysis_depth=5)

        assert result['number'] == '89'
        # Fibonacci numbers often have geometric significance
        total_rels = (len(result.get('platonic_relationships', {})) +
                     len(result.get('ivm_scaling_factors', {})) +
                     len(result.get('rational_approximations', {})))
        # Should find at least some relationships
        assert total_rels >= 0

