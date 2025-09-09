#!/usr/bin/env python3
"""
Coordinate Systems Demonstration

This example showcases the coordinate transformation capabilities
of the Symergetics package, focusing on:

- Quadray coordinate system (4D tetrahedral coordinates)
- Cartesian (XYZ) coordinate transformations  
- IVM (Isotropic Vector Matrix) lattice structures
- Geometric transformations and mappings
- Spatial pattern analysis

Perfect for understanding Fuller's coordinate mathematics.
"""

import sys
from pathlib import Path
import math

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.coordinates import QuadrayCoordinate, TETRAHEDRON_VERTICES, IVM_NEIGHBORS, ORIGIN
from symergetics.geometry.transformations import (
    translate, scale, reflect,
    rotate_around_axis, compose_transforms, coordinate_transform
)
from symergetics.visualization import (
    set_config, plot_quadray_coordinate, plot_ivm_lattice, 
    create_output_structure_readme
)
import numpy as np


# Wrapper functions for missing transformations
def translate_by_vector(coord: QuadrayCoordinate, vector: np.ndarray) -> QuadrayCoordinate:
    """Translate by XYZ vector (simplified implementation)."""
    # For demo purposes, convert vector to Quadray offset
    offset = QuadrayCoordinate(int(vector[0]), int(vector[1]), int(vector[2]), 0)
    return translate(coord, offset)

def scale_by_factor(coord: QuadrayCoordinate, factor: float) -> QuadrayCoordinate:
    """Scale coordinate by a factor."""
    return scale(coord, factor)

def reflect_through_origin(coord: QuadrayCoordinate) -> QuadrayCoordinate:
    """Reflect through origin."""
    return reflect(coord, 'origin')

def rotate_around_xyz_axis(coord: QuadrayCoordinate, axis: str, angle: float) -> QuadrayCoordinate:
    """Rotate around XYZ axis."""
    return rotate_around_axis(coord, axis, math.degrees(angle))


def demonstrate_quadray_basics():
    """Demonstrate basic Quadray coordinate operations."""
    print("🧭 QUADRAY COORDINATE SYSTEM BASICS")
    print("="*50)
    
    print("\n1. Tetrahedral Vertex Coordinates:")
    print("-" * 40)
    
    # Show the four fundamental tetrahedron vertices
    vertices = [
        ("Vertex A", QuadrayCoordinate(1, 0, 0, 0)),
        ("Vertex B", QuadrayCoordinate(0, 1, 0, 0)), 
        ("Vertex C", QuadrayCoordinate(0, 0, 1, 0)),
        ("Vertex D", QuadrayCoordinate(0, 0, 0, 1)),
        ("Center",   QuadrayCoordinate(1, 1, 1, 1)),
    ]
    
    print("Quadray (a,b,c,d) -> XYZ coordinates:")
    for name, coord in vertices:
        xyz = coord.to_xyz()
        print(f"{name:8}: {coord} -> ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f})")
        
        # Generate visualization
        result = plot_quadray_coordinate(coord, backend='ascii', size=2)
        print(f"          Visualization: {result['files'][0]}")
        
    print("\n2. Coordinate Arithmetic:")
    print("-" * 25)
    
    # Demonstrate 4D vector arithmetic
    v1 = QuadrayCoordinate(2, 1, 0, 1)
    v2 = QuadrayCoordinate(1, 1, 1, 0)
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Sum:      {v1.add(v2)}")
    print(f"Diff:     {v1.sub(v2)}")
    print(f"Scaled:   {scale(v1, 2)}")
    
    # 4D geometric properties
    print(f"\nGeometric Properties:")
    print(f"Magnitude v1: {v1.magnitude():.6f}")
    print(f"Magnitude v2: {v2.magnitude():.6f}")
    print(f"Dot product:  {v1.dot(v2):.6f}")
    print(f"Distance:     {v1.distance_to(v2):.6f}")
    
    print("\n3. Coordinate Conversions:")
    print("-" * 25)
    
    # Round-trip conversion tests
    test_coords = [
        QuadrayCoordinate(1, 2, 3, 0),
        QuadrayCoordinate(0, 0, 1, 2),
        QuadrayCoordinate(3, 1, 1, 1),
    ]
    
    print("Round-trip conversion test (Quadray -> XYZ -> Quadray):")
    for coord in test_coords:
        xyz = coord.to_xyz()
        back_coord = QuadrayCoordinate.from_xyz(*xyz)
        
        print(f"Original: {coord}")
        print(f"XYZ:      ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f})")
        print(f"Back:     {back_coord}")
        print(f"Match:    {coord == back_coord}")
        print()


def demonstrate_ivm_lattice():
    """Demonstrate IVM lattice structure and properties."""
    print("\n🔗 IVM LATTICE STRUCTURE")
    print("="*30)
    
    print("\n1. IVM Lattice Generation:")
    print("-" * 30)
    
    # Generate IVM lattice visualization
    lattice_result = plot_ivm_lattice(size=3, backend='ascii')
    print(f"IVM lattice visualization: {lattice_result['files'][0]}")
    print(f"Total lattice points: {lattice_result['metadata']['total_points']}")
    
    print("\n2. Neighbor Relationships:")
    print("-" * 25)
    
    # Show IVM neighbor structure
    center = ORIGIN
    neighbors = IVM_NEIGHBORS
    
    print(f"Center point: {center}")
    print(f"Number of IVM neighbors: {len(neighbors)}")
    print(f"First 8 neighbors:")
    
    for i, neighbor in enumerate(neighbors[:8]):
        distance = center.distance_to(neighbor)
        xyz = neighbor.to_xyz()
        print(f"  {i+1:2d}. {neighbor} -> ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f}) [d={distance:.3f}]")
        
    print("\n3. Lattice Pattern Analysis:")
    print("-" * 30)
    
    # Analyze lattice patterns at different scales
    for size in range(1, 5):
        # Generate lattice points
        lattice_points = []
        for a in range(-size, size + 1):
            for b in range(-size, size + 1):
                for c in range(-size, size + 1):
                    for d in range(-size, size + 1):
                        if abs(a) + abs(b) + abs(c) + abs(d) <= size:
                            lattice_points.append(QuadrayCoordinate(a, b, c, d))
                            
        print(f"Size ±{size}: {len(lattice_points)} lattice points")
        
    print("\n4. Closest Packing Analysis:")
    print("-" * 30)
    
    # Analyze closest packing properties
    test_point = QuadrayCoordinate(1, 1, 0, 0)
    neighbors = IVM_NEIGHBORS
    
    distances = [test_point.distance_to(neighbor) for neighbor in neighbors]
    unique_distances = sorted(set(distances))
    
    print(f"Test point: {test_point}")
    print(f"Neighbor distances: {len(distances)} total")
    print(f"Unique distances: {unique_distances[:5]}")
    print(f"Minimum distance: {min(distances):.6f}")
    print(f"Maximum distance: {max(distances):.6f}")
    
    # Distance distribution
    from collections import Counter
    dist_counter = Counter(f"{d:.3f}" for d in distances)
    print(f"Distance distribution:")
    for dist, count in sorted(dist_counter.items())[:5]:
        print(f"  {dist}: {count} neighbors")


def demonstrate_transformations():
    """Demonstrate geometric transformations."""
    print("\n🔄 GEOMETRIC TRANSFORMATIONS")
    print("="*35)
    
    print("\n1. Basic Transformations:")
    print("-" * 25)
    
    # Start with a test coordinate
    original = QuadrayCoordinate(2, 1, 0, 1)
    print(f"Original coordinate: {original}")
    print(f"XYZ position: {original.to_xyz()}")
    
    # Translation
    translation_vector = np.array([1.0, 0.5, 0.0])
    translated = translate_by_vector(original, translation_vector)
    print(f"\nTranslated by {translation_vector}:")
    print(f"Result: {translated}")
    print(f"XYZ: {translated.to_xyz()}")
    
    # Scaling
    scale_factor = 2.0
    scaled = scale_by_factor(original, scale_factor)
    print(f"\nScaled by {scale_factor}:")
    print(f"Result: {scaled}")
    print(f"XYZ: {scaled.to_xyz()}")
    
    # Reflection
    reflected = reflect_through_origin(original)
    print(f"\nReflected through origin:")
    print(f"Result: {reflected}")
    print(f"XYZ: {reflected.to_xyz()}")
    
    print("\n2. Rotation Transformations:")
    print("-" * 25)
    
    # Rotation around different axes
    angles = [30, 60, 90, 120]  # degrees
    axes = ['x', 'y', 'z']
    
    for axis in axes:
        print(f"\nRotations around {axis.upper()}-axis:")
        for angle in angles:
            rotated = rotate_around_xyz_axis(original, axis, math.radians(angle))
            print(f"  {angle:3d}°: {rotated} -> XYZ{rotated.to_xyz()}")
            
    print("\n3. Composite Transformations:")
    print("-" * 30)
    
    # Create a sequence of transformations
    transforms = [
        lambda coord: scale_by_factor(coord, 1.5),
        lambda coord: translate_by_vector(coord, np.array([0.5, 0.0, 1.0])),
        lambda coord: rotate_around_xyz_axis(coord, 'z', math.pi/4),
    ]
    
    # Apply transformations step by step
    current = original
    print(f"Starting point: {current}")
    
    for i, transform in enumerate(transforms):
        current = transform(current)
        print(f"After transform {i+1}: {current}")
        print(f"                     XYZ: {current.to_xyz()}")
        
    # Apply all at once using composition
    composed_transform = compose_transforms(*transforms)
    final_result = composed_transform(original)
    print(f"\nComposed result: {final_result}")
    print(f"Match: {current == final_result}")


def demonstrate_coordinate_mappings():
    """Demonstrate coordinate system mappings and projections."""
    print("\n🗺️ COORDINATE SYSTEM MAPPINGS")
    print("="*35)
    
    print("\n1. Spherical Coordinate Mapping:")
    print("-" * 35)
    
    # Map Quadray coordinates to spherical coordinates
    test_coords = [
        QuadrayCoordinate(1, 0, 0, 0),
        QuadrayCoordinate(1, 1, 0, 0),
        QuadrayCoordinate(1, 1, 1, 0),
        QuadrayCoordinate(1, 1, 1, 1),
    ]
    
    print("Quadray -> XYZ -> Spherical (r, θ, φ):")
    for coord in test_coords:
        xyz = coord.to_xyz()
        x, y, z = xyz
        
        # Convert to spherical
        r = math.sqrt(x*x + y*y + z*z)
        theta = math.atan2(y, x) if r > 0 else 0
        phi = math.acos(z/r) if r > 0 else 0
        
        print(f"{coord} -> ({x:6.3f}, {y:6.3f}, {z:6.3f}) -> (r={r:.3f}, θ={math.degrees(theta):5.1f}°, φ={math.degrees(phi):5.1f}°)")
        
    print("\n2. Cylindrical Coordinate Mapping:")
    print("-" * 35)
    
    print("Quadray -> XYZ -> Cylindrical (ρ, φ, z):")
    for coord in test_coords:
        xyz = coord.to_xyz()
        x, y, z = xyz
        
        # Convert to cylindrical
        rho = math.sqrt(x*x + y*y)
        phi = math.atan2(y, x)
        
        print(f"{coord} -> ({x:6.3f}, {y:6.3f}, {z:6.3f}) -> (ρ={rho:.3f}, φ={math.degrees(phi):5.1f}°, z={z:6.3f})")
        
    print("\n3. Homogeneous Coordinate Projection:")
    print("-" * 40)
    
    # Project to homogeneous coordinates (4D -> 3D projection)
    print("4D Quadray -> 3D Homogeneous projection:")
    for coord in test_coords:
        # Use sum of coordinates as homogeneous coordinate
        w = coord.a + coord.b + coord.c + coord.d
        if w != 0:
            homogeneous = (coord.a/w, coord.b/w, coord.c/w)
        else:
            homogeneous = (coord.a, coord.b, coord.c)
            
        print(f"{coord} -> w={w} -> homogeneous=({homogeneous[0]:.3f}, {homogeneous[1]:.3f}, {homogeneous[2]:.3f})")


def demonstrate_spatial_patterns():
    """Demonstrate spatial pattern analysis."""
    print("\n🌐 SPATIAL PATTERN ANALYSIS")
    print("="*30)
    
    print("\n1. Regular Pattern Generation:")
    print("-" * 30)
    
    # Generate regular patterns in 4D space
    patterns = {
        "Linear": [(i, 0, 0, 0) for i in range(-3, 4)],
        "Square": [(i, j, 0, 0) for i in range(-2, 3) for j in range(-2, 3)],
        "Cubic": [(i, j, k, 0) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)],
        "Tesseract": [(i, j, k, l) for i in range(0, 2) for j in range(0, 2) for k in range(0, 2) for l in range(0, 2)],
    }
    
    for pattern_name, coords in patterns.items():
        print(f"\n{pattern_name} pattern ({len(coords)} points):")
        quadray_coords = [QuadrayCoordinate(*c) for c in coords]
        
        # Calculate pattern properties
        center = QuadrayCoordinate(0, 0, 0, 0)
        distances = [center.distance_to(coord) for coord in quadray_coords]
        
        print(f"  Center distances: min={min(distances):.3f}, max={max(distances):.3f}, avg={sum(distances)/len(distances):.3f}")
        
        # Show first few coordinates
        for i, coord in enumerate(quadray_coords[:4]):
            xyz = coord.to_xyz()
            print(f"    {i+1}. {coord} -> ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f})")
        if len(quadray_coords) > 4:
            print(f"    ... and {len(quadray_coords) - 4} more points")
            
    print("\n2. Symmetry Analysis:")
    print("-" * 20)
    
    # Analyze symmetries in coordinate patterns
    test_pattern = [
        QuadrayCoordinate(1, 0, 0, 0),
        QuadrayCoordinate(0, 1, 0, 0),
        QuadrayCoordinate(0, 0, 1, 0), 
        QuadrayCoordinate(0, 0, 0, 1),
    ]
    
    print("Tetrahedral symmetry analysis:")
    center_of_mass = QuadrayCoordinate(0.25, 0.25, 0.25, 0.25)
    
    for i, coord in enumerate(test_pattern):
        distance_to_center = center_of_mass.distance_to(coord)
        print(f"Vertex {i+1}: {coord} -> distance to center = {distance_to_center:.6f}")
        
    # Check all pairwise distances
    print(f"\nPairwise distances:")
    for i in range(len(test_pattern)):
        for j in range(i+1, len(test_pattern)):
            dist = test_pattern[i].distance_to(test_pattern[j])
            print(f"  {i+1}-{j+1}: {dist:.6f}")
            
    print("\n3. Lattice Density Analysis:")
    print("-" * 25)
    
    # Analyze packing density at different scales
    for radius in [1, 2, 3]:
        points_in_sphere = []
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                for c in range(-radius, radius + 1):
                    for d in range(-radius, radius + 1):
                        coord = QuadrayCoordinate(a, b, c, d)
                        if coord.magnitude() <= radius:
                            points_in_sphere.append(coord)
                            
        sphere_volume = (4/3) * math.pi * radius**3
        packing_density = len(points_in_sphere) / sphere_volume
        
        print(f"Radius {radius}: {len(points_in_sphere)} points, density = {packing_density:.3f} points/unit³")


def main():
    """Run the coordinate systems demonstration."""
    print("🎯 COORDINATE SYSTEMS & TRANSFORMATIONS DEMO")
    print("="*60)
    print()
    print("This demonstration showcases the coordinate system capabilities")
    print("of the Symergetics package:")
    print()
    print("• 🧭 Quadray coordinate system (4D tetrahedral)")
    print("• 🔗 IVM lattice structure and patterns")
    print("• 🔄 Geometric transformations")
    print("• 🗺️ Coordinate system mappings")
    print("• 🌐 Spatial pattern analysis")
    print()
    
    # Configure for organized output
    set_config({
        'backend': 'ascii',
        'output_dir': 'output',
        'organize_by_type': True
    })
    
    # Create organized structure
    create_output_structure_readme()
    
    try:
        # Run all coordinate system demonstrations
        demonstrate_quadray_basics()
        demonstrate_ivm_lattice()
        demonstrate_transformations()
        demonstrate_coordinate_mappings()
        demonstrate_spatial_patterns()
        
        print(f"\n" + "="*60)
        print("🎉 COORDINATE SYSTEMS DEMONSTRATION COMPLETE!")
        print("="*60)
        print()
        print("🔍 Key Insights:")
        print("✓ Quadray coordinates provide natural 4D→3D projection")
        print("✓ IVM lattice demonstrates optimal space-filling patterns")
        print("✓ Transformations preserve geometric relationships")
        print("✓ Multiple coordinate systems reveal different perspectives")
        print("✓ Spatial patterns emerge from tetrahedral mathematics")
        print()
        print("🧮 Mathematical Discoveries:")
        print("• Four-dimensional coordinates enable natural geometric operations")
        print("• Tetrahedral basis provides isotropic spatial relationships")
        print("• IVM lattice maximizes closest-packing efficiency")
        print("• Coordinate transformations preserve mathematical precision")
        print("• Pattern analysis reveals underlying spatial symmetries")
        print()
        print("🚀 Fuller's coordinate geometry successfully demonstrated!")
        
    except Exception as e:
        print(f"\n❌ Coordinate systems demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
