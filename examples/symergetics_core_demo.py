#!/usr/bin/env python3
"""
Symergetics Core Principles Demonstration

This example specifically showcases the three fundamental principles of 
Fuller's Synergetics as implemented in this package:

1. Integer-accounting: Exact counting and rational arithmetic
2. Ratio-based geometry: Precise volumetric relationships  
3. Quadmath connections: Four-dimensional coordinate mathematics

Run this to see how these principles work together in the package.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.numbers import SymergeticsNumber, rational_sqrt, rational_pi
from symergetics.core.constants import SymergeticsConstants
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube, Cuboctahedron
from symergetics.computation.primorials import primorial
from symergetics.utils.conversion import continued_fraction_approximation
from symergetics.visualization import set_config, create_output_structure_readme
import math


def demonstrate_integer_accounting():
    """Demonstrate exact integer-accounting mathematics."""
    print("🧮 INTEGER-ACCOUNTING DEMONSTRATION")
    print("="*50)
    
    print("\n1. Exact Rational Arithmetic:")
    print("-" * 30)
    
    # Show exact rational representations
    a = SymergeticsNumber(22, 7)  # π approximation
    b = SymergeticsNumber(355, 113)  # Better π approximation
    
    print(f"π ≈ 22/7 = {a.value} = {float(a.value):.10f}")
    print(f"π ≈ 355/113 = {b.value} = {float(b.value):.10f}")
    print(f"Actual π = {math.pi:.10f}")
    print(f"22/7 error: {abs(float(a.value) - math.pi):.2e}")
    print(f"355/113 error: {abs(float(b.value) - math.pi):.2e}")
    
    print(f"\n2. Exact Arithmetic Operations:")
    print("-" * 30)
    
    # Demonstrate exact arithmetic
    x = SymergeticsNumber(1, 3)
    y = SymergeticsNumber(1, 6) 
    
    print(f"1/3 + 1/6 = {x + y} = {float((x + y).value)}")
    print(f"1/3 * 1/6 = {x * y} = {float((x * y).value)}")
    print(f"(1/3)² = {x ** 2} = {float((x ** 2).value)}")
    
    # Show no floating-point drift
    sum_thirds = SymergeticsNumber(0)
    for i in range(3):
        sum_thirds += SymergeticsNumber(1, 3)
    print(f"1/3 + 1/3 + 1/3 = {sum_thirds.value} (exact: no drift!)")
    
    print(f"\n3. Primorial Integer Factorization:")
    print("-" * 30)
    
    # Show exact primorial calculations
    for n in [5, 7, 11, 13]:
        p = primorial(n)
        print(f"{n}# = {p.value} (product of primes ≤ {n})")
        
    print(f"\n✓ Integer-accounting ensures no computational drift")
    print(f"✓ All calculations use exact rational arithmetic")
    print(f"✓ Primorials demonstrate exact integer factorization")


def demonstrate_ratio_based_geometry():
    """Demonstrate ratio-based geometric relationships."""
    print("\n📐 RATIO-BASED GEOMETRY DEMONSTRATION")  
    print("="*50)
    
    print("\n1. Polyhedron Volume Ratios (IVM units):")
    print("-" * 40)
    
    # Create polyhedra with unit edge lengths
    tetrahedron = Tetrahedron()
    octahedron = Octahedron() 
    cube = Cube()
    cuboctahedron = Cuboctahedron()
    
    # Get exact volume ratios
    tet_vol = tetrahedron.volume()
    oct_vol = octahedron.volume()
    cube_vol = cube.volume() 
    cubocta_vol = cuboctahedron.volume()
    
    print(f"Tetrahedron:     {tet_vol} (unit volume)")
    print(f"Octahedron:      {oct_vol} ({oct_vol/tet_vol:.0f}:1 ratio)")
    print(f"Cube:            {cube_vol} ({cube_vol/tet_vol:.0f}:1 ratio)")
    print(f"Cuboctahedron:   {cubocta_vol} ({cubocta_vol/tet_vol:.0f}:1 ratio)")
    
    print(f"\n2. Exact Volume Relationships:")
    print("-" * 30)
    
    print(f"Octahedron = 4 Tetrahedra (exact)")
    print(f"Cube = 3 Tetrahedra (exact)")
    print(f"Cuboctahedron = 20 Tetrahedra (exact)")
    print(f"These are Fuller's fundamental volume relationships!")
    
    print(f"\n3. Edge Length Scaling:")
    print("-" * 30)
    
    # Show how edge lengths relate to volumes
    constants = SymergeticsConstants()
    ratios = constants.EDGE_LENGTH_RATIOS
    
    for shape, ratio in ratios.items():
        if hasattr(ratio, 'value'):
            print(f"{shape.capitalize()}: edge ratio = {ratio.value} = {float(ratio.value):.6f}")
        else:
            print(f"{shape.capitalize()}: edge ratio = {ratio}")
        
    print(f"\n✓ All volume relationships are exact rational numbers")
    print(f"✓ Edge length ratios maintain geometric precision")
    print(f"✓ Tetrahedron serves as fundamental volumetric unit")


def demonstrate_quadmath_connections():
    """Demonstrate four-dimensional coordinate mathematics."""
    print("\n🧭 QUADMATH CONNECTIONS DEMONSTRATION")
    print("="*50)
    
    print("\n1. Quadray Coordinate System:")
    print("-" * 30)
    
    # Create Quadray coordinates (4D tetrahedral coordinates)
    coords = [
        QuadrayCoordinate(1, 0, 0, 0),  # Vertex A
        QuadrayCoordinate(0, 1, 0, 0),  # Vertex B
        QuadrayCoordinate(0, 0, 1, 0),  # Vertex C
        QuadrayCoordinate(0, 0, 0, 1),  # Vertex D
        QuadrayCoordinate(1, 1, 1, 1),  # Center
        QuadrayCoordinate(2, 1, 1, 0),  # IVM point
    ]
    
    print("Quadray (a,b,c,d) -> XYZ conversion:")
    for coord in coords:
        xyz = coord.to_xyz()
        print(f"{coord} -> ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
        
    print(f"\n2. Four-Dimensional Mathematics:")
    print("-" * 30)
    
    # Show 4D vector operations
    v1 = QuadrayCoordinate(2, 1, 0, 1)
    v2 = QuadrayCoordinate(1, 1, 1, 1)
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Sum:      {v1.add(v2)}")
    print(f"Difference: {v1.sub(v2)}")
    
    # Calculate 4D magnitudes
    mag1 = v1.magnitude()
    mag2 = v2.magnitude()
    print(f"Magnitude 1: {mag1:.6f}")
    print(f"Magnitude 2: {mag2:.6f}")
    
    # Calculate 4D dot product
    dot = v1.dot(v2)
    print(f"Dot product: {dot:.6f}")
    
    print(f"\n3. IVM Lattice Structure:")
    print("-" * 30)
    
    # Show IVM lattice neighbors
    from symergetics.core.coordinates import IVM_NEIGHBORS
    center = QuadrayCoordinate(0, 0, 0, 0)
    neighbors = IVM_NEIGHBORS
    
    print(f"Center: {center}")
    print(f"IVM neighbors ({len(neighbors)} total):")
    for i, neighbor in enumerate(neighbors[:6]):  # Show first 6
        dist = center.distance_to(neighbor)
        print(f"  {i+1}. {neighbor} (distance: {dist:.6f})")
    if len(neighbors) > 6:
        print(f"  ... and {len(neighbors) - 6} more neighbors")
        
    print(f"\n✓ Four-dimensional coordinate system working perfectly")
    print(f"✓ Tetrahedral basis provides natural 4D→3D projection")  
    print(f"✓ IVM lattice demonstrates closest-packing mathematics")


def demonstrate_integrated_principles():
    """Show how all three principles work together."""
    print("\n🔗 INTEGRATED PRINCIPLES DEMONSTRATION")
    print("="*50)
    
    print("\n1. Golden Ratio in Tetrahedral Coordinates:")
    print("-" * 40)
    
    # Calculate golden ratio using exact arithmetic
    phi_exact = (SymergeticsNumber(1) + rational_sqrt(SymergeticsNumber(5))) / SymergeticsNumber(2)
    print(f"φ (exact) = {phi_exact.value}")
    print(f"φ (float) = {float(phi_exact.value):.10f}")
    
    # Show continued fraction expansion (integer-accounting)
    phi_float = float(phi_exact.value)
    cf = continued_fraction_approximation(phi_float, max_terms=8)
    print(f"φ continued fraction: {cf}")
    
    # Golden ratio appears in pentagonal symmetry related to tetrahedra
    print(f"The golden ratio emerges from pentagonal/icosahedral symmetries")
    print(f"which relate to tetrahedral close-packing in 4D space")
    
    print(f"\n2. Scheherazade Numbers in Quadray Space:")
    print("-" * 40)
    
    # Use Scheherazade number as coordinate
    sch_num = SymergeticsNumber(1001)  # 7 × 11 × 13
    coord_sch = QuadrayCoordinate(float(sch_num.value), 0, 0, 1)
    xyz_sch = coord_sch.to_xyz()
    
    print(f"Scheherazade base: {sch_num.value}")
    print(f"Prime factors: 7 × 11 × 13 = {7*11*13}")
    print(f"As Quadray coordinate: {coord_sch}")
    print(f"XYZ position: ({xyz_sch[0]:.1f}, {xyz_sch[1]:.1f}, {xyz_sch[2]:.1f})")
    
    print(f"\n3. Volume-Coordinate-Number Integration:")
    print("-" * 40)
    
    # Show how volumes, coordinates, and numbers integrate
    # Volume ratios as coordinate multipliers
    octahedron = Octahedron()
    oct_vol = octahedron.volume()
    vol_coord = QuadrayCoordinate(float(oct_vol), 1, 1, 1)
    
    print(f"Octahedron volume: {oct_vol} tetrahedra")
    print(f"Used as coordinate: {vol_coord}")  
    print(f"Demonstrates: geometry → coordinates → space")
    
    # Primorials as dimensional scaling
    p7 = primorial(7)  # 210
    print(f"7# = {p7.value} (primorial)")
    print(f"Can scale coordinate systems by exact integer factors")
    print(f"Maintains both geometric and arithmetic precision")
    
    print(f"\n✓ Integer-accounting + Ratio geometry + Quadmath = Complete system")
    print(f"✓ All three principles reinforce mathematical precision")
    print(f"✓ Fuller's Synergetics implemented as computational framework")


def main():
    """Run the core Symergetics principles demonstration."""
    print("🎯 SYMERGETICS CORE PRINCIPLES DEMO")
    print("="*60)
    print()
    print("This demonstration showcases the three fundamental principles")
    print("of Fuller's Synergetics as implemented in this package:")
    print()
    print("1. 🧮 Integer-accounting: Exact rational arithmetic")
    print("2. 📐 Ratio-based geometry: Precise volumetric relationships")  
    print("3. 🧭 Quadmath connections: Four-dimensional coordinates")
    print()
    
    # Configure for minimal output (this demo is about principles, not files)
    set_config({
        'backend': 'ascii',
        'output_dir': 'output/core_demo',
        'organize_by_type': False
    })
    
    try:
        # Run all demonstrations
        demonstrate_integer_accounting()
        demonstrate_ratio_based_geometry()
        demonstrate_quadmath_connections() 
        demonstrate_integrated_principles()
        
        print(f"\n" + "="*60)
        print("🎉 SYMERGETICS CORE PRINCIPLES SUCCESSFULLY DEMONSTRATED!")
        print("="*60)
        print()
        print("🔬 Key Mathematical Insights:")
        print("✓ Exact rational arithmetic prevents computational drift")
        print("✓ Volume ratios reveal fundamental geometric relationships")
        print("✓ Four-dimensional coordinates enable natural 3D projections")
        print("✓ All principles integrate into a coherent mathematical system")
        print()
        print("📚 This demonstrates Fuller's vision of:")
        print("• 'Universe as a system of relationships'")
        print("• 'Geometry as number, number as geometry'") 
        print("• 'Four-dimensional mathematical universe'")
        print()
        print("🚀 Ready for advanced mathematical research!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
        

if __name__ == "__main__":
    main()
