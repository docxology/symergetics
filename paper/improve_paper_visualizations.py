#!/usr/bin/env python3
"""
Enhanced Paper Visualizations Generator

This script generates improved visualizations for the Symergetics paper with:
- Enhanced panels and multi-panel layouts
- Better legends and annotations
- Improved font sizes and readability
- Professional scientific formatting
- Consistent styling across all figures

Author: Daniel Ari Friedman
Email: daniel@activeinference.institute
ORCID: 0000-0001-6232-9096
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube, Cuboctahedron
from symergetics.utils.conversion import continued_fraction_approximation, convergents_from_continued_fraction
from symergetics.computation.palindromes import analyze_scheherazade_ssrcd

# Set up enhanced styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Enhanced configuration
ENHANCED_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': {
        'title': 16,
        'subtitle': 14,
        'label': 12,
        'tick': 10,
        'legend': 11,
        'annotation': 10
    },
    'colors': {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D',
        'background': '#F8F9FA',
        'grid': '#E9ECEF',
        'text': '#212529'
    },
    'line_width': 2.5,
    'marker_size': 8,
    'alpha': 0.8
}

def setup_enhanced_plot():
    """Set up matplotlib with enhanced styling."""
    plt.rcParams.update({
        'font.size': ENHANCED_CONFIG['font_size']['label'],
        'axes.titlesize': ENHANCED_CONFIG['font_size']['title'],
        'axes.labelsize': ENHANCED_CONFIG['font_size']['label'],
        'xtick.labelsize': ENHANCED_CONFIG['font_size']['tick'],
        'ytick.labelsize': ENHANCED_CONFIG['font_size']['tick'],
        'legend.fontsize': ENHANCED_CONFIG['font_size']['legend'],
        'figure.titlesize': ENHANCED_CONFIG['font_size']['title'],
        'lines.linewidth': ENHANCED_CONFIG['line_width'],
        'lines.markersize': ENHANCED_CONFIG['marker_size'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': ENHANCED_CONFIG['colors']['background'],
        'figure.facecolor': 'white',
        'axes.edgecolor': ENHANCED_CONFIG['colors']['text'],
        'text.color': ENHANCED_CONFIG['colors']['text']
    })

def create_enhanced_quadray_visualization():
    """Create enhanced Quadray coordinate visualizations with multiple panels."""
    print("Creating enhanced Quadray coordinate visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Origin coordinate (0,0,0,0)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    coord1 = QuadrayCoordinate(0, 0, 0, 0)
    _plot_quadray_coordinate_enhanced(ax1, coord1, "Origin (0,0,0,0)")
    
    # Panel 2: Advanced coordinate (2,1,1,0)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    coord2 = QuadrayCoordinate(2, 1, 1, 0)
    _plot_quadray_coordinate_enhanced(ax2, coord2, "Advanced (2,1,1,0)")
    
    # Panel 3: Complex coordinate (3,2,1,0) with multiple points
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    _plot_complex_quadray_coordinates(ax3, "Complex Multi-Point Analysis")
    
    # Panel 4: Coordinate system overview
    ax4 = fig.add_subplot(gs[1, :])
    _plot_quadray_system_overview(ax4)
    
    # Panel 5: Transformation equations
    ax5 = fig.add_subplot(gs[2, :])
    _plot_transformation_equations(ax5)
    
    # Panel 6: IVM lattice visualization
    ax6 = fig.add_subplot(gs[3, :])
    _plot_ivm_lattice_analysis(ax6)
    
    plt.suptitle('Comprehensive Quadray Coordinate System Analysis', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "geometric" / "coordinates"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual panels
    fig.savefig(output_path / "quadray_coordinate_enhanced_multi_panel.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Also save the individual coordinates as referenced in the paper
    fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
    _plot_quadray_coordinate_enhanced(ax1, coord1, "Quadray Coordinate System Origin")
    fig1.savefig(output_path / "quadray_coordinate_0_0_0_0.png", 
                 dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig1)
    
    # Create a much more complex advanced visualization
    fig2, ax2 = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
    _plot_advanced_quadray_analysis(ax2, "Advanced Quadray Coordinate Analysis")
    fig2.savefig(output_path / "quadray_coordinate_2_1_1_0.png", 
                 dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig2)
    
    plt.close(fig)
    print(f"✅ Enhanced Quadray visualizations saved to {output_path}")

def _plot_quadray_coordinate_enhanced(ax, coord, title):
    """Plot enhanced Quadray coordinate with better styling."""
    # Convert coordinate to XYZ
    x, y, z = coord.to_xyz()
    
    # Plot the main coordinate
    ax.scatter([x], [y], [z], color=ENHANCED_CONFIG['colors']['primary'],
              s=200, alpha=1.0, label=f'Coordinate ({coord.a},{coord.b},{coord.c},{coord.d})')
    
    # Add direction vectors for quadray axes
    quadray_directions = [
        QuadrayCoordinate(1, 0, 0, 0),  # A-direction
        QuadrayCoordinate(0, 1, 0, 0),  # B-direction
        QuadrayCoordinate(0, 0, 1, 0),  # C-direction
        QuadrayCoordinate(0, 0, 0, 1),  # D-direction
    ]
    
    direction_colors = [ENHANCED_CONFIG['colors']['secondary'], 
                       ENHANCED_CONFIG['colors']['accent'], 
                       ENHANCED_CONFIG['colors']['success'], 
                       '#FF6B35']
    direction_labels = ['A-axis', 'B-axis', 'C-axis', 'D-axis']
    
    vector_scale = 0.8
    for i, (direction, color, label) in enumerate(zip(quadray_directions, direction_colors, direction_labels)):
        dir_xyz = direction.to_xyz()
        scaled_dir = np.array(dir_xyz) * vector_scale
        ax.quiver(0, 0, 0, scaled_dir[0], scaled_dir[1], scaled_dir[2],
                 color=color, alpha=0.7, linewidth=3, arrow_length_ratio=0.1,
                 label=label)
    
    # Enhanced styling
    ax.set_xlabel('X', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Y', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_zlabel('Z', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_title(title, fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    
    # Set equal aspect ratio
    max_range = 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Enhanced legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=ENHANCED_CONFIG['font_size']['legend'])
    ax.grid(True, alpha=0.3)

def _plot_quadray_system_overview(ax):
    """Plot overview of Quadray coordinate system properties."""
    ax.text(0.5, 0.9, 'Quadray Coordinate System Properties', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    # Properties text
    properties = [
        "• Four-dimensional tetrahedral coordinate system",
        "• Constraint: a + b + c + d = 0 (normalization)",
        "• At least one coordinate is zero after normalization",
        "• Natural representation for tetrahedral geometry",
        "• Exact conversion to/from Cartesian coordinates",
        "• Preserves spatial relationships with mathematical precision"
    ]
    
    for i, prop in enumerate(properties):
        ax.text(0.05, 0.7 - i*0.1, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_transformation_equations(ax):
    """Plot transformation equations between Quadray and Cartesian coordinates."""
    ax.text(0.5, 0.9, 'Coordinate Transformation Equations', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    # Transformation equations
    equations = [
        "Quadray to Cartesian:",
        "x = (a - b) / √2",
        "y = (a + b - 2c) / √6",
        "z = (a + b + c - 3d) / √12",
        "",
        "Cartesian to Quadray:",
        "a = (x + y/√3 + z/√6) / √2",
        "b = (-x + y/√3 + z/√6) / √2", 
        "c = (-2y/√3 + z/√6) / √2",
        "d = (-3z/√6) / √2"
    ]
    
    for i, eq in enumerate(equations):
        if eq.startswith("Quadray") or eq.startswith("Cartesian"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.5, 0.7 - i*0.08, eq, ha='center', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_complex_quadray_coordinates(ax, title):
    """Plot complex multi-point Quadray coordinate analysis."""
    # Generate multiple coordinates for analysis
    coords = [
        QuadrayCoordinate(3, 2, 1, 0),
        QuadrayCoordinate(2, 2, 0, 0),
        QuadrayCoordinate(1, 1, 1, 0),
        QuadrayCoordinate(4, 1, 1, 0),
        QuadrayCoordinate(2, 3, 1, 0)
    ]
    
    colors = [ENHANCED_CONFIG['colors']['primary'], 
              ENHANCED_CONFIG['colors']['secondary'],
              ENHANCED_CONFIG['colors']['accent'],
              ENHANCED_CONFIG['colors']['success'],
              '#FF6B35']
    
    for i, (coord, color) in enumerate(zip(coords, colors)):
        x, y, z = coord.to_xyz()
        ax.scatter([x], [y], [z], color=color, s=150, alpha=0.8, 
                  label=f'({coord.a},{coord.b},{coord.c},{coord.d})')
    
    # Add connecting lines to show relationships
    for i in range(len(coords)-1):
        x1, y1, z1 = coords[i].to_xyz()
        x2, y2, z2 = coords[i+1].to_xyz()
        ax.plot([x1, x2], [y1, y2], [z1, z2], 
               color=ENHANCED_CONFIG['colors']['text'], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Y', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_zlabel('Z', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_title(title, fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    
    # Set equal aspect ratio
    max_range = 2.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
              fontsize=ENHANCED_CONFIG['font_size']['legend'])
    ax.grid(True, alpha=0.3)

def _plot_advanced_quadray_analysis(ax, title):
    """Plot advanced Quadray coordinate analysis with multiple features."""
    # Generate a grid of coordinates
    coords = []
    for a in range(0, 4):
        for b in range(0, 4):
            for c in range(0, 4):
                d = -(a + b + c)  # Ensure sum is zero
                if d >= 0 and d <= 3:
                    coords.append(QuadrayCoordinate(a, b, c, d))
    
    # Plot all coordinates
    x_vals, y_vals, z_vals = [], [], []
    for coord in coords:
        x, y, z = coord.to_xyz()
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
    
    # Create a scatter plot with color mapping based on coordinate values
    scatter = ax.scatter(x_vals, y_vals, z_vals, 
                        c=range(len(coords)), cmap='viridis', 
                        s=100, alpha=0.7)
    
    # Add specific highlighted points
    highlight_coords = [
        QuadrayCoordinate(0, 0, 0, 0),
        QuadrayCoordinate(2, 1, 1, 0),
        QuadrayCoordinate(3, 2, 1, 0)
    ]
    
    for coord in highlight_coords:
        x, y, z = coord.to_xyz()
        ax.scatter([x], [y], [z], color='red', s=200, alpha=1.0, 
                  marker='*', edgecolors='black', linewidth=2)
    
    # Add tetrahedral structure
    tetra_vertices = [
        QuadrayCoordinate(1, 0, 0, 0),
        QuadrayCoordinate(0, 1, 0, 0),
        QuadrayCoordinate(0, 0, 1, 0),
        QuadrayCoordinate(0, 0, 0, 1)
    ]
    
    tetra_x = [coord.to_xyz()[0] for coord in tetra_vertices]
    tetra_y = [coord.to_xyz()[1] for coord in tetra_vertices]
    tetra_z = [coord.to_xyz()[2] for coord in tetra_vertices]
    
    # Draw tetrahedron edges
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for edge in edges:
        ax.plot([tetra_x[edge[0]], tetra_x[edge[1]]], 
               [tetra_y[edge[0]], tetra_y[edge[1]]], 
               [tetra_z[edge[0]], tetra_z[edge[1]]], 
               color='blue', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('X', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Y', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_zlabel('Z', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_title(title, fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    
    # Set equal aspect ratio
    max_range = 2.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Coordinate Index', fontsize=ENHANCED_CONFIG['font_size']['label'])

def _plot_ivm_lattice_analysis(ax):
    """Plot IVM lattice analysis and properties."""
    ax.text(0.5, 0.9, 'Isotropic Vector Matrix (IVM) Lattice Analysis', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    # IVM properties and analysis
    properties = [
        "• Lattice Structure: Tetrahedral close-packing arrangement",
        "• Fundamental Unit: Tetrahedron (1 IVM unit volume)",
        "• Volume Relationships:",
        "  - Octahedron: 4 IVM units (4 × tetrahedron)",
        "  - Cube: 3 IVM units (3 × tetrahedron)", 
        "  - Cuboctahedron: 20 IVM units (20 × tetrahedron)",
        "• Coordinate System: Four-dimensional tetrahedral coordinates",
        "• Mathematical Precision: Exact rational arithmetic throughout",
        "• Applications: Crystallography, materials science, geometric analysis",
        "• Symmetry: Tetrahedral symmetry group operations",
        "• Density: Optimal sphere packing in 3D space",
        "• Geometric Properties: All angles and distances exactly calculable"
    ]
    
    for i, prop in enumerate(properties):
        if prop.startswith("• Lattice") or prop.startswith("• Volume") or prop.startswith("• Coordinate"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.06, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def create_enhanced_polyhedra_visualizations():
    """Create enhanced 3D polyhedra visualizations with better rendering."""
    print("Creating enhanced polyhedra visualizations...")
    
    polyhedra = [
        ('tetrahedron', Tetrahedron()),
        ('cube', Cube()),
        ('octahedron', Octahedron()),
        ('cuboctahedron', Cuboctahedron())
    ]
    
    output_path = project_root / "output" / "geometric" / "polyhedra"
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, poly in polyhedra:
        # Create enhanced 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        _plot_enhanced_polyhedron(ax, poly, name)
        
        # Save enhanced version
        filename = f"{name}_3d_enhanced.png"
        fig.savefig(output_path / filename, dpi=ENHANCED_CONFIG['dpi'], 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close(fig)
        print(f"✅ Enhanced {name} visualization saved")
    
    # Create multi-panel comparison
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, (name, poly) in enumerate(polyhedra):
        ax = fig.add_subplot(gs[i//2, i%2], projection='3d')
        _plot_enhanced_polyhedron(ax, poly, name, show_legend=False)
    
    plt.suptitle('Enhanced Platonic Solids Visualization', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    fig.savefig(output_path / "platonic_solids_enhanced_comparison.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    print("✅ Enhanced polyhedra visualizations completed")

def _plot_enhanced_polyhedron(ax, polyhedron, name, show_legend=True):
    """Plot enhanced polyhedron with better rendering."""
    # Get vertex coordinates
    vertices_xyz = polyhedron.to_xyz_vertices()
    if not vertices_xyz:
        return
    
    vertices = np.array(vertices_xyz)
    
    # Enhanced surface rendering with gradient colors
    try:
        faces = polyhedron.faces()
        if faces:
            face_colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
            for i, face in enumerate(faces):
                face_vertices = np.array([v.to_xyz() for v in face])
                if len(face_vertices) >= 3:
                    face_collection = Poly3DCollection([face_vertices],
                                                     alpha=0.7,
                                                     facecolors=[face_colors[i]],
                                                     edgecolors=ENHANCED_CONFIG['colors']['text'],
                                                     linewidths=1.5)
                    ax.add_collection3d(face_collection)
    except:
        pass
    
    # Enhanced wireframe
    try:
        faces = polyhedron.faces()
        if faces:
            for face in faces:
                face_vertices = np.array([v.to_xyz() for v in face])
                for i in range(len(face_vertices)):
                    start = face_vertices[i]
                    end = face_vertices[(i + 1) % len(face_vertices)]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                           color=ENHANCED_CONFIG['colors']['text'], linewidth=2, alpha=0.8)
    except:
        pass
    
    # Enhanced vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              color=ENHANCED_CONFIG['colors']['primary'], s=100, alpha=1.0)
    
    # Enhanced labels and styling
    ax.set_xlabel('X', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Y', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_zlabel('Z', fontsize=ENHANCED_CONFIG['font_size']['label'])
    
    title = f'{name.title()} - Volume: {polyhedron.volume()} IVM units'
    ax.set_title(title, fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
                fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid_x = vertices[:, 0].mean()
    mid_y = vertices[:, 1].mean()
    mid_z = vertices[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Enhanced grid and background
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(ENHANCED_CONFIG['colors']['background'])
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)

def create_enhanced_mathematical_visualizations():
    """Create enhanced mathematical visualizations with better panels."""
    print("Creating enhanced mathematical visualizations...")
    
    # Enhanced continued fraction visualization
    _create_enhanced_continued_fraction_visualization()
    
    # Enhanced base conversion visualization
    _create_enhanced_base_conversion_visualization()
    
    # Enhanced pattern discovery visualization
    _create_enhanced_pattern_discovery_visualization()
    
    # Enhanced palindrome analysis visualization
    _create_enhanced_palindrome_analysis_visualization()
    
    # Enhanced Scheherazade number analysis
    _create_enhanced_scheherazade_analysis_visualization()
    
    # Enhanced primorial sequence visualization
    _create_enhanced_primorial_visualization()
    
    print("✅ Enhanced mathematical visualizations completed")

def _create_enhanced_continued_fraction_visualization():
    """Create enhanced continued fraction convergence visualization."""
    value = 3.14159
    max_terms = 15
    
    # Get continued fraction data
    terms = continued_fraction_approximation(value, max_terms)
    convergents = convergents_from_continued_fraction(terms)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Terms as bars
    ax1 = fig.add_subplot(gs[0, 0])
    positions = list(range(len(terms)))
    bars = ax1.bar(positions, terms, color=ENHANCED_CONFIG['colors']['primary'], alpha=0.7)
    ax1.set_title('Continued Fraction Terms', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax1.set_xlabel('Term Position', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax1.set_ylabel('Term Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax1.set_xticks(positions)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, term in zip(bars, terms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{term}', ha='center', va='bottom', fontsize=ENHANCED_CONFIG['font_size']['tick'])
    
    # Panel 2: Convergence plot
    ax2 = fig.add_subplot(gs[0, 1])
    if convergents:
        approximations = [num/den for num, den in convergents]
        errors = [abs(approx - value) for approx in approximations]
        
        ax2.semilogy(range(len(errors)), errors, 'o-', 
                    color=ENHANCED_CONFIG['colors']['secondary'], 
                    linewidth=ENHANCED_CONFIG['line_width'], 
                    markersize=ENHANCED_CONFIG['marker_size'])
        ax2.set_title('Convergence to True Value (Log Scale)', 
                     fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=ENHANCED_CONFIG['font_size']['label'])
        ax2.set_ylabel('Absolute Error', fontsize=ENHANCED_CONFIG['font_size']['label'])
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Fraction values over iterations
    ax3 = fig.add_subplot(gs[1, :])
    if convergents:
        x_vals = range(len(convergents))
        y_vals = [num/den for num, den in convergents]
        
        ax3.plot(x_vals, y_vals, 'o-', color=ENHANCED_CONFIG['colors']['accent'], 
                linewidth=ENHANCED_CONFIG['line_width'], markersize=ENHANCED_CONFIG['marker_size'])
        ax3.axhline(y=value, color=ENHANCED_CONFIG['colors']['success'], 
                   linestyle='--', linewidth=2, alpha=0.8, label=f'True Value: {value}')
        ax3.set_title('Convergent Values Over Iterations', 
                     fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
        ax3.set_xlabel('Iteration', fontsize=ENHANCED_CONFIG['font_size']['label'])
        ax3.set_ylabel('Approximation Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
        ax3.legend(fontsize=ENHANCED_CONFIG['font_size']['legend'])
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Mathematical properties
    ax4 = fig.add_subplot(gs[2, :])
    ax4.text(0.5, 0.9, 'Mathematical Properties of Continued Fractions', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax4.transAxes)
    
    properties = [
        f"• Target Value: {value}",
        f"• Number of Terms: {len(terms)}",
        f"• Final Approximation: {convergents[-1][0]/convergents[-1][1]:.10f}" if convergents else "• No convergents",
        f"• Final Error: {abs(convergents[-1][0]/convergents[-1][1] - value):.2e}" if convergents else "• No convergents",
        "• Each convergent is the best rational approximation",
        "• Error decreases exponentially with each term"
    ]
    
    for i, prop in enumerate(properties):
        ax4.text(0.05, 0.7 - i*0.1, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.suptitle('Enhanced Continued Fraction Convergence Analysis', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "continued_fractions"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "continued_fraction_convergence_3_14159_15.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _create_enhanced_base_conversion_visualization():
    """Create enhanced base conversion visualization."""
    number = 30030  # 6th primorial
    base = 2
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Base conversion process
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_base_conversion_process(ax1, number, base)
    
    # Panel 2: Binary representation
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_binary_representation(ax2, number)
    
    # Panel 3: Mathematical properties
    ax3 = fig.add_subplot(gs[1, :])
    _plot_primorial_properties(ax3, number)
    
    plt.suptitle('Enhanced Base Conversion Analysis for Primorial Number', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "base_conversions"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "base_conversion_30030_base_10_to_2.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _plot_base_conversion_process(ax, number, base):
    """Plot the base conversion process."""
    # Convert number to different bases
    bases = [2, 8, 10, 16]
    representations = []
    
    for b in bases:
        if b == 10:
            representations.append(str(number))
        else:
            representations.append(format(number, f'0{len(format(number, "b"))}b') if b == 2 else 
                                 format(number, f'0{len(format(number, "o"))}o') if b == 8 else 
                                 format(number, f'0{len(format(number, "x"))}x'))
    
    # Create bar plot
    x_pos = np.arange(len(bases))
    bars = ax.bar(x_pos, [len(rep) for rep in representations], 
                 color=[ENHANCED_CONFIG['colors']['primary'], 
                       ENHANCED_CONFIG['colors']['secondary'],
                       ENHANCED_CONFIG['colors']['accent'],
                       ENHANCED_CONFIG['colors']['success']], alpha=0.7)
    
    ax.set_title('Number Length in Different Bases', 
                fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Base', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Number of Digits', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Base {b}' for b in bases])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rep in zip(bars, representations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{len(rep)}', ha='center', va='bottom', 
                fontsize=ENHANCED_CONFIG['font_size']['tick'])

def _plot_binary_representation(ax, number):
    """Plot binary representation with bit patterns."""
    binary = format(number, 'b')
    
    # Create bit visualization
    bits = [int(bit) for bit in binary]
    x_pos = np.arange(len(bits))
    
    colors = [ENHANCED_CONFIG['colors']['primary'] if bit == 1 else ENHANCED_CONFIG['colors']['grid'] 
             for bit in bits]
    
    bars = ax.bar(x_pos, bits, color=colors, alpha=0.8)
    ax.set_title(f'Binary Representation: {number}', 
                fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Bit Position', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Bit Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax.grid(True, alpha=0.3)
    
    # Add bit labels
    for i, (bar, bit) in enumerate(zip(bars, bits)):
        if i % max(1, len(bits)//10) == 0:  # Show every 10th bit
            ax.text(bar.get_x() + bar.get_width()/2., bit + 0.1,
                   f'{bit}', ha='center', va='bottom', 
                   fontsize=ENHANCED_CONFIG['font_size']['tick'])

def _plot_primorial_properties(ax, number):
    """Plot primorial number properties."""
    ax.text(0.5, 0.9, f'Primorial Number Properties: {number}', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    # Calculate some properties
    binary = format(number, 'b')
    hex_val = format(number, 'x')
    
    properties = [
        f"• Decimal: {number:,}",
        f"• Binary: {binary} ({len(binary)} bits)",
        f"• Hexadecimal: {hex_val.upper()}",
        f"• Prime factors: 2 × 3 × 5 × 7 × 11 × 13",
        f"• Number of 1-bits: {binary.count('1')}",
        f"• Number of 0-bits: {binary.count('0')}",
        f"• Bit length: {len(binary)}",
        f"• Is even: {number % 2 == 0}",
        f"• Divisible by 3: {number % 3 == 0}",
        f"• Divisible by 5: {number % 5 == 0}"
    ]
    
    for i, prop in enumerate(properties):
        ax.text(0.05, 0.7 - i*0.08, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _create_enhanced_pattern_discovery_visualization():
    """Create enhanced pattern discovery visualization."""
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Scheherazade number analysis
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_scheherazade_analysis(ax1)
    
    # Panel 2: Palindrome patterns
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_palindrome_patterns(ax2)
    
    # Panel 3: Mathematical patterns overview
    ax3 = fig.add_subplot(gs[1, :])
    _plot_mathematical_patterns_overview(ax3)
    
    # Panel 4: Pattern discovery algorithms
    ax4 = fig.add_subplot(gs[2, :])
    _plot_pattern_discovery_algorithms(ax4)
    
    plt.suptitle('Enhanced Pattern Discovery Analysis', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "pattern_discovery"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "pattern_discovery_geometric_pattern_discovery_analysis.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _plot_scheherazade_analysis(ax):
    """Plot Scheherazade number analysis."""
    # Analyze Scheherazade numbers (powers of 1001)
    powers = [1, 2, 3, 4, 5]
    scheherazade_numbers = [1001**p for p in powers]
    
    ax.plot(powers, scheherazade_numbers, 'o-', 
           color=ENHANCED_CONFIG['colors']['primary'], 
           linewidth=ENHANCED_CONFIG['line_width'], 
           markersize=ENHANCED_CONFIG['marker_size'])
    ax.set_title('Scheherazade Numbers (1001^n)', 
                fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Power (n)', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (p, val) in enumerate(zip(powers, scheherazade_numbers)):
        ax.text(p, val*1.1, f'{val:,}', ha='center', va='bottom', 
               fontsize=ENHANCED_CONFIG['font_size']['tick'], rotation=45)

def _plot_palindrome_patterns(ax):
    """Plot palindrome pattern analysis."""
    # Sample palindrome analysis
    numbers = [11, 22, 121, 131, 1331, 1001, 1221, 13331]
    is_palindrome = [str(n) == str(n)[::-1] for n in numbers]
    
    colors = [ENHANCED_CONFIG['colors']['success'] if p else ENHANCED_CONFIG['colors']['grid'] 
             for p in is_palindrome]
    
    bars = ax.bar(range(len(numbers)), [1 if p else 0 for p in is_palindrome], 
                 color=colors, alpha=0.7)
    ax.set_title('Palindrome Detection Analysis', 
                fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Number', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Is Palindrome', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(range(len(numbers)))
    ax.set_xticklabels([str(n) for n in numbers], rotation=45)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    
    # Add labels
    for i, (bar, num, is_pal) in enumerate(zip(bars, numbers, is_palindrome)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
               '✓' if is_pal else '✗', ha='center', va='bottom', 
               fontsize=ENHANCED_CONFIG['font_size']['tick'])

def _plot_mathematical_patterns_overview(ax):
    """Plot overview of mathematical patterns."""
    ax.text(0.5, 0.9, 'Mathematical Pattern Discovery Framework', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    patterns = [
        "• Scheherazade Numbers: Powers of 1001 with embedded patterns",
        "• Primorial Sequences: Cumulative products of prime numbers",
        "• Palindrome Detection: Multi-base symmetry analysis",
        "• Geometric Patterns: Spatial relationships in coordinate systems",
        "• Continued Fractions: Rational approximations of irrational numbers",
        "• Base Conversions: Number representation analysis",
        "• Pattern Recognition: Automated discovery of mathematical structures",
        "• Exact Arithmetic: Precision-preserving mathematical operations"
    ]
    
    for i, pattern in enumerate(patterns):
        ax.text(0.05, 0.7 - i*0.08, pattern, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_pattern_discovery_algorithms(ax):
    """Plot pattern discovery algorithms overview."""
    ax.text(0.5, 0.9, 'Pattern Discovery Algorithms', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    algorithms = [
        "• Exact Rational Arithmetic: Maintains precision in all calculations",
        "• Coordinate Transformations: Preserves geometric relationships exactly",
        "• Volume Calculations: Exact IVM unit computations for Platonic solids",
        "• Pattern Recognition: Multi-scale analysis of mathematical structures",
        "• Sequence Analysis: Growth rate and convergence pattern detection",
        "• Geometric Analysis: Spatial relationship and symmetry identification",
        "• Number Theory: Prime factorization and divisibility analysis",
        "• Visualization: High-quality rendering of mathematical concepts"
    ]
    
    for i, algo in enumerate(algorithms):
        ax.text(0.05, 0.7 - i*0.08, algo, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _create_enhanced_palindrome_analysis_visualization():
    """Create enhanced palindrome analysis visualization based on the report."""
    print("Creating enhanced palindrome analysis visualization...")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Palindrome pattern analysis for 121
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_palindrome_121_analysis(ax1)
    
    # Panel 2: Multi-base palindrome detection
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_multi_base_palindrome_analysis(ax2)
    
    # Panel 3: Palindrome density analysis
    ax3 = fig.add_subplot(gs[1, :])
    _plot_palindrome_density_analysis(ax3)
    
    # Panel 4: Statistical analysis of palindromes
    ax4 = fig.add_subplot(gs[2, :])
    _plot_palindrome_statistical_analysis(ax4)
    
    plt.suptitle('Enhanced Palindrome Pattern Analysis', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "palindromes"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "palindrome_analysis_enhanced.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _plot_palindrome_121_analysis(ax):
    """Plot detailed analysis of palindrome 121."""
    # Based on the ASCII report for 121
    digits = [1, 2, 1]
    positions = [0, 1, 2]
    
    # Create bar chart showing digit pattern
    bars = ax.bar(positions, digits, color=[ENHANCED_CONFIG['colors']['success'] if i == 1 else 
                                           ENHANCED_CONFIG['colors']['primary'] for i in digits], 
                 alpha=0.7)
    
    # Add digit labels
    for i, (pos, digit) in enumerate(zip(positions, digits)):
        ax.text(pos, digit + 0.1, str(digit), ha='center', va='bottom', 
               fontsize=ENHANCED_CONFIG['font_size']['tick'], fontweight='bold')
    
    ax.set_title('Palindrome 121 Analysis', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Digit Position', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Digit Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(positions)
    ax.set_xticklabels(['Left', 'Center', 'Right'])
    ax.grid(True, alpha=0.3)
    
    # Add symmetry indicators
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(1, max(digits) * 0.8, 'Symmetry Axis', ha='center', va='center', 
           fontsize=ENHANCED_CONFIG['font_size']['tick'], color='red', fontweight='bold')

def _plot_multi_base_palindrome_analysis(ax):
    """Plot multi-base palindrome analysis."""
    numbers = [121, 1331, 1001, 1221, 13331]
    bases = [2, 8, 10, 16]
    
    # Create heatmap of palindrome properties
    palindrome_matrix = []
    for num in numbers:
        row = []
        for base in bases:
            if base == 10:
                is_pal = str(num) == str(num)[::-1]
            else:
                # Convert to base and check palindrome
                converted = format(num, f'0{len(format(num, "b"))}b') if base == 2 else \
                           format(num, f'0{len(format(num, "o"))}o') if base == 8 else \
                           format(num, f'0{len(format(num, "x"))}x')
                is_pal = converted == converted[::-1]
            row.append(1 if is_pal else 0)
        palindrome_matrix.append(row)
    
    im = ax.imshow(palindrome_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_title('Multi-Base Palindrome Detection', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Number Base', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Number', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([f'Base {b}' for b in bases])
    ax.set_yticks(range(len(numbers)))
    ax.set_yticklabels([str(n) for n in numbers])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Is Palindrome', fontsize=ENHANCED_CONFIG['font_size']['label'])

def _plot_palindrome_density_analysis(ax):
    """Plot palindrome density analysis."""
    # Sample data for palindrome density
    ranges = ['1-100', '101-1000', '1001-10000', '10001-100000']
    densities = [9.0, 1.0, 0.1, 0.01]  # Approximate palindrome densities
    
    bars = ax.bar(ranges, densities, color=ENHANCED_CONFIG['colors']['primary'], alpha=0.7)
    ax.set_title('Palindrome Density by Number Range', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Number Range', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Palindrome Density (%)', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{density}%', ha='center', va='bottom', 
                fontsize=ENHANCED_CONFIG['font_size']['tick'])

def _plot_palindrome_statistical_analysis(ax):
    """Plot statistical analysis of palindromes."""
    ax.text(0.5, 0.9, 'Palindrome Statistical Analysis', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    # Statistical properties from the report
    properties = [
        "• Number: 121",
        "• Length: 3 digits",
        "• Is Palindromic: True",
        "• Palindromic Density: 33.333%",
        "• Symmetric Patterns: 1 (central_symmetry)",
        "• Digit Frequency Analysis:",
        "  - Digit 1: 2 occurrences (66.7%)",
        "  - Digit 2: 1 occurrence (33.3%)",
        "• Statistical Metrics:",
        "  - Entropy: 0.918 bits",
        "  - Assessment: Low entropy - highly structured",
        "  - Unique digits used: 2/10 possible digits",
        "• Pattern Characteristics:",
        "  - Central symmetry with single axis",
        "  - High digit repetition (digit 1 appears twice)",
        "  - Structured rather than random distribution"
    ]
    
    for i, prop in enumerate(properties):
        if prop.startswith("• Number") or prop.startswith("• Statistical") or prop.startswith("• Pattern"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.05, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _create_enhanced_scheherazade_analysis_visualization():
    """Create enhanced Scheherazade number analysis visualization."""
    print("Creating enhanced Scheherazade analysis visualization...")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Scheherazade number 1001^2 analysis
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_scheherazade_1001_squared_analysis(ax1)
    
    # Panel 2: Digit frequency analysis
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_scheherazade_digit_frequency(ax2)
    
    # Panel 3: Pattern analysis overview
    ax3 = fig.add_subplot(gs[1, :])
    _plot_scheherazade_pattern_analysis(ax3)
    
    # Panel 4: Mathematical properties
    ax4 = fig.add_subplot(gs[2, :])
    _plot_scheherazade_mathematical_properties(ax4)
    
    plt.suptitle('Enhanced Scheherazade Number Analysis (1001²)', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "scheherazade"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "scheherazade_analysis_enhanced.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _plot_scheherazade_1001_squared_analysis(ax):
    """Plot analysis of 1001^2 = 1002001."""
    # Based on the ASCII report
    number = 1002001
    digits = [int(d) for d in str(number)]
    positions = list(range(len(digits)))
    
    # Create bar chart
    colors = [ENHANCED_CONFIG['colors']['primary'] if d == 0 else 
              ENHANCED_CONFIG['colors']['secondary'] if d == 1 else
              ENHANCED_CONFIG['colors']['accent'] for d in digits]
    
    bars = ax.bar(positions, digits, color=colors, alpha=0.7)
    
    # Add digit labels
    for i, (pos, digit) in enumerate(zip(positions, digits)):
        ax.text(pos, digit + 0.1, str(digit), ha='center', va='bottom', 
               fontsize=ENHANCED_CONFIG['font_size']['tick'], fontweight='bold')
    
    ax.set_title('1001² = 1,002,001 Digit Analysis', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Digit Position', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Digit Value', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in positions])
    ax.grid(True, alpha=0.3)

def _plot_scheherazade_digit_frequency(ax):
    """Plot digit frequency analysis for Scheherazade numbers."""
    # Based on the report data
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    frequencies = [4, 2, 1, 0, 0, 0, 0, 0, 0, 0]  # From 1002001
    percentages = [57.1, 28.6, 14.3, 0, 0, 0, 0, 0, 0, 0]
    
    bars = ax.bar(digits, frequencies, color=ENHANCED_CONFIG['colors']['primary'], alpha=0.7)
    ax.set_title('Digit Frequency in 1001²', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('Digit', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Frequency', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_xticks(digits)
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, freq, pct in zip(bars, frequencies, percentages):
        if freq > 0:
            ax.text(bar.get_x() + bar.get_width()/2., freq + 0.1,
                    f'{pct}%', ha='center', va='bottom', 
                    fontsize=ENHANCED_CONFIG['font_size']['tick'])

def _plot_scheherazade_pattern_analysis(ax):
    """Plot pattern analysis for Scheherazade numbers."""
    ax.text(0.5, 0.9, 'Scheherazade Number Pattern Analysis', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    patterns = [
        "• Number: 1,002,001 (1001²)",
        "• Length: 7 digits",
        "• Is Palindromic: True",
        "• Palindromic Density: 42.857%",
        "• Pattern Analysis Results:",
        "  - Palindromic Patterns: 3",
        "  - Symmetric Patterns: 1",
        "  - Repeated Digit Patterns: 5",
        "• Digit Distribution:",
        "  - Digit 0: 4 occurrences (57.1%)",
        "  - Digit 1: 2 occurrences (28.6%)",
        "  - Digit 2: 1 occurrence (14.3%)",
        "• Statistical Properties:",
        "  - Entropy: 1.379 bits (low entropy)",
        "  - Assessment: Highly structured digit patterns",
        "  - Symmetry: Central palindrome structure"
    ]
    
    for i, pattern in enumerate(patterns):
        if pattern.startswith("• Number") or pattern.startswith("• Pattern") or pattern.startswith("• Statistical"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.05, pattern, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_scheherazade_mathematical_properties(ax):
    """Plot mathematical properties of Scheherazade numbers."""
    ax.text(0.5, 0.9, 'Mathematical Properties of Scheherazade Numbers', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    properties = [
        "• Definition: Powers of 1001 (10³ + 1)",
        "• Factorization: 1001 = 7 × 11 × 13",
        "• Pattern Discovery:",
        "  - Palindromic sequences in specific positions",
        "  - Pascal triangle coefficients embedded naturally",
        "  - Prime factor relationships follow geometric progressions",
        "• Mathematical Significance:",
        "  - Reveals embedded patterns through exact arithmetic",
        "  - Demonstrates deep mathematical structures",
        "  - Shows connections between number theory and geometry",
        "• Computational Analysis:",
        "  - Requires exact precision to uncover patterns",
        "  - Floating-point approximations obscure structures",
        "  - Exact rational arithmetic reveals hidden relationships"
    ]
    
    for i, prop in enumerate(properties):
        if prop.startswith("• Definition") or prop.startswith("• Pattern") or prop.startswith("• Mathematical"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.05, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _create_enhanced_primorial_visualization():
    """Create enhanced primorial sequence visualization."""
    print("Creating enhanced primorial visualization...")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Primorial sequence growth
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_primorial_sequence_growth(ax1)
    
    # Panel 2: Prime factor accumulation
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_prime_factor_accumulation(ax2)
    
    # Panel 3: Primorial properties analysis
    ax3 = fig.add_subplot(gs[1, :])
    _plot_primorial_properties_analysis(ax3)
    
    # Panel 4: Mathematical significance
    ax4 = fig.add_subplot(gs[2, :])
    _plot_primorial_mathematical_significance(ax4)
    
    plt.suptitle('Enhanced Primorial Sequence Analysis', 
                 fontsize=ENHANCED_CONFIG['font_size']['title'], 
                 fontweight='bold', y=0.95)
    
    # Save the enhanced visualization
    output_path = project_root / "output" / "mathematical" / "primorials"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / "primorial_analysis_enhanced.png", 
                dpi=ENHANCED_CONFIG['dpi'], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def _plot_primorial_sequence_growth(ax):
    """Plot primorial sequence growth."""
    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
    
    ax.semilogy(n_values, primorials, 'o-', color=ENHANCED_CONFIG['colors']['primary'], 
               linewidth=ENHANCED_CONFIG['line_width'], markersize=ENHANCED_CONFIG['marker_size'])
    ax.set_title('Primorial Sequence Growth', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('n (number of primes)', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('n# (log scale)', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.grid(True, alpha=0.3)
    
    # Add value labels for key points
    for i, (n, p) in enumerate(zip(n_values, primorials)):
        if i % 2 == 0:  # Show every other point
            ax.text(n, p * 1.2, f'{p:,}', ha='center', va='bottom', 
                   fontsize=ENHANCED_CONFIG['font_size']['tick'], rotation=45)

def _plot_prime_factor_accumulation(ax):
    """Plot prime factor accumulation in primorials."""
    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    prime_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    digit_counts = [1, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    
    ax.plot(n_values, prime_counts, 'o-', color=ENHANCED_CONFIG['colors']['primary'], 
           label='Number of Prime Factors', linewidth=ENHANCED_CONFIG['line_width'])
    ax.plot(n_values, digit_counts, 's-', color=ENHANCED_CONFIG['colors']['secondary'], 
           label='Number of Digits', linewidth=ENHANCED_CONFIG['line_width'])
    
    ax.set_title('Prime Factor Accumulation', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], fontweight='bold')
    ax.set_xlabel('n (number of primes)', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.set_ylabel('Count', fontsize=ENHANCED_CONFIG['font_size']['label'])
    ax.legend(fontsize=ENHANCED_CONFIG['font_size']['legend'])
    ax.grid(True, alpha=0.3)

def _plot_primorial_properties_analysis(ax):
    """Plot primorial properties analysis."""
    ax.text(0.5, 0.9, 'Primorial Sequence Properties Analysis', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    properties = [
        "• Definition: n# = product of first n prime numbers",
        "• Examples:",
        "  - 1# = 2",
        "  - 2# = 2 × 3 = 6", 
        "  - 3# = 2 × 3 × 5 = 30",
        "  - 6# = 2 × 3 × 5 × 7 × 11 × 13 = 30,030",
        "• Growth Characteristics:",
        "  - Exponential growth rate",
        "  - Each term multiplies by next prime",
        "  - Rapid increase in digit count",
        "• Mathematical Significance:",
        "  - Applications in number theory",
        "  - Connections to Riemann zeta function",
        "  - Prime number distribution analysis",
        "• Computational Requirements:",
        "  - Exact arithmetic essential for accuracy",
        "  - Large number handling capabilities",
        "  - Efficient prime generation algorithms"
    ]
    
    for i, prop in enumerate(properties):
        if prop.startswith("• Definition") or prop.startswith("• Growth") or prop.startswith("• Mathematical"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.05, prop, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_primorial_mathematical_significance(ax):
    """Plot mathematical significance of primorial sequences."""
    ax.text(0.5, 0.9, 'Mathematical Significance of Primorial Sequences', 
            ha='center', va='center', fontsize=ENHANCED_CONFIG['font_size']['subtitle'], 
            fontweight='bold', transform=ax.transAxes)
    
    significance = [
        "• Number Theory Applications:",
        "  - Study of prime number distribution",
        "  - Analysis of prime gaps and patterns",
        "  - Investigation of twin prime conjectures",
        "• Advanced Mathematics:",
        "  - Connections to Riemann zeta function",
        "  - Applications in analytic number theory",
        "  - Role in prime counting functions",
        "• Computational Mathematics:",
        "  - Testing exact arithmetic capabilities",
        "  - Benchmarking large number operations",
        "  - Validation of mathematical algorithms",
        "• Research Applications:",
        "  - Cryptography and security analysis",
        "  - Random number generation",
        "  - Mathematical pattern discovery"
    ]
    
    for i, item in enumerate(significance):
        if item.startswith("• Number") or item.startswith("• Advanced") or item.startswith("• Computational"):
            fontweight = 'bold'
            color = ENHANCED_CONFIG['colors']['primary']
        else:
            fontweight = 'normal'
            color = ENHANCED_CONFIG['colors']['text']
        
        ax.text(0.05, 0.8 - i*0.05, item, ha='left', va='center', 
                fontsize=ENHANCED_CONFIG['font_size']['label'], 
                fontweight=fontweight, color=color, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def main():
    """Main function to generate all enhanced visualizations."""
    print("="*60)
    print("ENHANCED PAPER VISUALIZATIONS GENERATOR")
    print("="*60)
    print("Generating improved visualizations with:")
    print("• Enhanced panels and multi-panel layouts")
    print("• Better legends and annotations")
    print("• Improved font sizes and readability")
    print("• Professional scientific formatting")
    print("• Consistent styling across all figures")
    print()
    
    # Set up enhanced plotting
    setup_enhanced_plot()
    
    try:
        # Generate enhanced visualizations
        create_enhanced_quadray_visualization()
        create_enhanced_polyhedra_visualizations()
        create_enhanced_mathematical_visualizations()
        
        print("\n" + "="*60)
        print("✅ ALL ENHANCED VISUALIZATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Enhanced visualizations saved to:")
        print(f"• Geometric: {project_root}/output/geometric/")
        print(f"• Mathematical: {project_root}/output/mathematical/")
        print("\nAll figures now feature:")
        print("• Professional scientific formatting")
        print("• Enhanced readability and clarity")
        print("• Consistent styling and color schemes")
        print("• Multi-panel layouts where appropriate")
        print("• Improved legends and annotations")
        
    except Exception as e:
        print(f"\n❌ Error generating enhanced visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
