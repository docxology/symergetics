"""
Symergetics Constants and Mathematical Values

This module contains all the fundamental constants used in Fuller's Synergetics,
including volume ratios, cosmic scaling factors, and special numbers.

Key Features:
- Exact rational representations of all constants
- Volume ratios for regular polyhedra
- Cosmic hierarchy scaling relationships
- Scheherazade numbers and primorials

Author: Symergetics Team
"""

from fractions import Fraction
from typing import Dict, List, Any, Union
import math
from .numbers import SymergeticsNumber


class SymergeticsConstants:
    """
    Container for all mathematical constants used in Synergetics.

    Provides exact rational representations of geometric ratios, cosmic scaling
    factors, and special numbers from Fuller's work.
    """

    # Volume Ratios (exact rational values)
    VOLUME_RATIOS = {
        'tetrahedron': SymergeticsNumber(1),      # Unit volume
        'octahedron': SymergeticsNumber(4),       # 4 tetrahedra
        'cube': SymergeticsNumber(3),             # 3 tetrahedra
        'cuboctahedron': SymergeticsNumber(20),   # 20 tetrahedra
        'rhombic_dodecahedron': SymergeticsNumber(6),  # 6 tetrahedra
        'rhombic_triacontahedron': SymergeticsNumber(120),  # 120 tetrahedra
    }

    # Edge length ratios for unit volume normalization
    EDGE_LENGTH_RATIOS = {
        'tetrahedron': SymergeticsNumber(1),
        'octahedron': SymergeticsNumber.sqrt(SymergeticsNumber(2)),  # √2
        'cube': SymergeticsNumber.sqrt(SymergeticsNumber(3)),        # √3
        'cuboctahedron': SymergeticsNumber(2),      # 2
    }

    # Scheherazade numbers (powers of 1001 = 7 × 11 × 13)
    SCHEHERAZADE_BASE = SymergeticsNumber(1001)
    SCHEHERAZADE_POWERS = {
        1: SymergeticsNumber(1001),
        2: SymergeticsNumber(1002001),        # 1001²
        3: SymergeticsNumber(1003003001),     # 1001³
        4: SymergeticsNumber(1004006004001),  # 1001⁴
        5: SymergeticsNumber(10050010005001), # 1001⁵
        6: SymergeticsNumber(10060015006001), # 1001⁶ = 1,006,015,020,015,006,001
    }

    # Primorials (product of primes ≤ n)
    PRIMORIALS = {
        2: SymergeticsNumber(2),
        3: SymergeticsNumber(6),        # 2 × 3
        5: SymergeticsNumber(30),       # 2 × 3 × 5
        7: SymergeticsNumber(210),      # 2 × 3 × 5 × 7
        11: SymergeticsNumber(2310),    # 2 × 3 × 5 × 7 × 11
        13: SymergeticsNumber(30030),   # 2 × 3 × 5 × 7 × 11 × 13
        17: SymergeticsNumber(510510),  # ... × 17
        19: SymergeticsNumber(9699690), # ... × 19
    }

    # Fuller's 14-illion cosmic abundance number
    # 2¹² × 3⁸ × 5⁶ × 7⁶ × 11⁶ × 13⁶ × 17² × 19 × 23 × 29 × 31 × 37 × 41 × 43
    COSMIC_ABUNDANCE_14_ILLION = (
        SymergeticsNumber(2)**12 *
        SymergeticsNumber(3)**8 *
        SymergeticsNumber(5)**6 *
        SymergeticsNumber(7)**6 *
        SymergeticsNumber(11)**6 *
        SymergeticsNumber(13)**6 *
        SymergeticsNumber(17)**2 *
        SymergeticsNumber(19) *
        SymergeticsNumber(23) *
        SymergeticsNumber(29) *
        SymergeticsNumber(31) *
        SymergeticsNumber(37) *
        SymergeticsNumber(41) *
        SymergeticsNumber(43)
    )

    # Cosmic scaling relationships (exact values)
    COSMIC_SCALING = {
        'atomic_diameters_per_inch': SymergeticsNumber(25_000_000_000),  # 2.5 × 10¹⁰
        'earth_circumference_seconds': SymergeticsNumber(1_296_000),      # 360° × 3600 seconds
        'earth_circumference_degrees': SymergeticsNumber(360),
        'light_year_inches': SymergeticsNumber(9_460_730_472_580_800),  # Approximate
        'universal_vector_equilibrium': SymergeticsNumber(12),           # 12 around 1
    }

    # Vector Equilibrium (Cuboctahedron) constants
    VECTOR_EQUILIBRIUM = {
        'frequency_formula': SymergeticsNumber(12),  # f² × 10 + 2
        'surface_vectors': SymergeticsNumber(12),
        'edge_vectors': SymergeticsNumber(24),
        'vertex_vectors': SymergeticsNumber(12),
        'tetrahedral_cells': SymergeticsNumber(8),
        'octahedral_cells': SymergeticsNumber(6),
    }

    # Special irrational approximations (with high precision rationals)
    IRRATIONAL_APPROXIMATIONS = {
        'pi': SymergeticsNumber(Fraction(math.pi).limit_denominator(1000000)),
        'e': SymergeticsNumber(Fraction(math.e).limit_denominator(1000000)),
        'phi': SymergeticsNumber(Fraction((1 + math.sqrt(5)) / 2).limit_denominator(1000000)),  # Golden ratio
        'sqrt2': SymergeticsNumber.sqrt(SymergeticsNumber(2), 1000000),
        'sqrt3': SymergeticsNumber.sqrt(SymergeticsNumber(3), 1000000),
    }

    # Geodesic constants
    GEODESIC = {
        'icosahedron_edges': SymergeticsNumber(30),
        'dodecahedron_faces': SymergeticsNumber(12),
        'frequency_1_class_1': SymergeticsNumber(1),  # Icosahedron
        'frequency_1_class_2': SymergeticsNumber(2),  # Class II subdivision
    }

    @classmethod
    def get_volume_ratio(cls, polyhedron: str) -> SymergeticsNumber:
        """
        Get the volume ratio for a regular polyhedron.

        Args:
            polyhedron: Name of the polyhedron ('tetrahedron', 'octahedron', etc.)

        Returns:
            SymergeticsNumber: Volume ratio in tetrahedra

        Raises:
            ValueError: If polyhedron name is not recognized
        """
        if polyhedron not in cls.VOLUME_RATIOS:
            available = list(cls.VOLUME_RATIOS.keys())
            raise ValueError(f"Unknown polyhedron '{polyhedron}'. Available: {available}")
        return cls.VOLUME_RATIOS[polyhedron]

    @classmethod
    def get_scheherazade_power(cls, n: int) -> SymergeticsNumber:
        """
        Calculate or retrieve the nth power of 1001.

        Args:
            n: Power to calculate (1001^n)

        Returns:
            SymergeticsNumber: The nth Scheherazade number
        """
        if n in cls.SCHEHERAZADE_POWERS:
            return cls.SCHEHERAZADE_POWERS[n]

        # Calculate on demand for higher powers
        result = cls.SCHEHERAZADE_BASE ** n
        cls.SCHEHERAZADE_POWERS[n] = result
        return result

    @classmethod
    def get_primorial(cls, n: int) -> SymergeticsNumber:
        """
        Get the primorial of n (product of primes ≤ n).

        Args:
            n: Upper limit for prime product

        Returns:
            SymergeticsNumber: The primorial value
        """
        if n in cls.PRIMORIALS:
            return cls.PRIMORIALS[n]

        # Calculate on demand
        primes = cls._get_primes_up_to(n)
        result = SymergeticsNumber(1)
        for prime in primes:
            result = result * SymergeticsNumber(prime)

        cls.PRIMORIALS[n] = result
        return result

    @classmethod
    def _get_primes_up_to(cls, n: int) -> List[int]:
        """Get all prime numbers up to n using Sieve of Eratosthenes."""
        if n < 2:
            return []

        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False

        return [i for i in range(2, n + 1) if sieve[i]]

    @classmethod
    def get_cosmic_scale_factor(cls, from_unit: str, to_unit: str) -> SymergeticsNumber:
        """
        Get scaling factor between cosmic units.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            SymergeticsNumber: Scaling factor
        """
        # Define conversion factors (simplified examples)
        conversions = {
            ('inches', 'atomic_diameters'): cls.COSMIC_SCALING['atomic_diameters_per_inch'],
            ('earth_circumference', 'degrees'): cls.COSMIC_SCALING['earth_circumference_degrees'],
            ('earth_circumference', 'seconds'): cls.COSMIC_SCALING['earth_circumference_seconds'],
        }

        key = (from_unit, to_unit)
        if key in conversions:
            return conversions[key]

        # Try reverse conversion
        reverse_key = (to_unit, from_unit)
        if reverse_key in conversions:
            return SymergeticsNumber(1) / conversions[reverse_key]

        raise ValueError(f"No conversion factor available for {from_unit} to {to_unit}")

    @classmethod
    def cosmic_abundance_factors(cls) -> Dict[str, Union[int, List[int], SymergeticsNumber]]:
        """
        Calculate the factors of Fuller's 14-illion cosmic abundance number.

        Returns:
            Dict: Dictionary containing the prime factors and final number
        """
        from ..computation.primorials import cosmic_abundance_factors
        return cosmic_abundance_factors()

    @classmethod
    def all_constants(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all constants organized by category.

        Returns:
            Dict[str, Dict[str, Any]]: All constants grouped by category
        """
        return {
            'volume_ratios': {k: float(v.value) for k, v in cls.VOLUME_RATIOS.items()},
            'scheherazade_powers': {k: str(v.value) for k, v in cls.SCHEHERAZADE_POWERS.items()},
            'primorials': {k: str(v.value) for k, v in cls.PRIMORIALS.items()},
            'cosmic_scaling': {k: str(v.value) for k, v in cls.COSMIC_SCALING.items()},
            'irrational_approximations': {k: float(v.value) for k, v in cls.IRRATIONAL_APPROXIMATIONS.items()},
        }

    @classmethod
    def by_category(cls, category: str) -> Dict[str, Any]:
        """
        Get constants for a specific category.

        Args:
            category: Category name ('volume', 'scheherazade', 'primorial', etc.)

        Returns:
            Dict[str, Any]: Constants for the specified category
        """
        all_cats = cls.all_constants()

        # Map common category names to internal categories
        category_mapping = {
            'volume': 'volume_ratios',
            'volumes': 'volume_ratios',
            'scheherazade': 'scheherazade_powers',
            'scheherazade_numbers': 'scheherazade_powers',
            'primorial': 'primorials',
            'primorials': 'primorials',
            'cosmic': 'cosmic_scaling',
            'cosmic_scaling': 'cosmic_scaling',
            'irrational': 'irrational_approximations',
            'irrationals': 'irrational_approximations',
        }

        internal_cat = category_mapping.get(category.lower(), category.lower())

        if internal_cat not in all_cats:
            available = list(all_cats.keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available}")

        return all_cats[internal_cat]


# Convenience constants for direct access
import math

# Common constants
PHI = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['phi']  # Golden ratio
PI = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['pi']    # π
E = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['e']      # Euler's number
SQRT2 = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['sqrt2']  # √2
SQRT3 = SymergeticsConstants.IRRATIONAL_APPROXIMATIONS['sqrt3']  # √3

# Synergetics-specific constants
SCHEHERAZADE_BASE = SymergeticsConstants.SCHEHERAZADE_BASE
COSMIC_ABUNDANCE = SymergeticsConstants.COSMIC_ABUNDANCE_14_ILLION
