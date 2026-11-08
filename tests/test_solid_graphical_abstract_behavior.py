"""Real solid projections, bounded exact displays, section transitions and ownership."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from symergetics.core.coordinates import QuadrayCoordinate as Q
from symergetics.geometry.polyhedra import Cube, Cuboctahedron, Octahedron, Tetrahedron
from symergetics.visualization import set_config
from symergetics.visualization.geometry import (
    _draw_polyhedron_projection,
    _polyhedron_frame,
    plot_polyhedron_graphical_abstract,
)


@pytest.fixture(autouse=True)
def isolated_output(tmp_path):
    set_config(dict(output_dir=str(tmp_path), organize_by_type=False,
                    include_timestamps=False, dpi=50, formats=['svg']))


def svg_result(model, **kwargs):
    plt = pytest.importorskip('matplotlib.pyplot')
    figures = plt.get_fignums()
    with plt.rc_context({'svg.fonttype': 'none'}):
        result = plot_polyhedron_graphical_abstract(model, **kwargs)
    assert plt.get_fignums() == figures
    return result, ET.parse(result['files'][0]).getroot()


@pytest.mark.parametrize('factory', [Tetrahedron, Cube, Octahedron, Cuboctahedron])
def test_projection_contains_every_declared_edge_without_triangle_fan_diagonals(factory):
    plt = pytest.importorskip('matplotlib.pyplot')
    model = factory()
    # Deformation must not change connectivity or introduce a projected hull.
    model.vertices[0] = model.vertices[0].add(Q(3, 0, 0, 0))
    vertices, _ = _polyhedron_frame(model)
    expected = sorted({tuple(sorted((face[i], face[(i + 1) % len(face)])))
                       for face in model.face_indices() for i in range(len(face))})
    fig, ax = plt.subplots()
    try:
        _draw_polyhedron_projection(ax, model, vertices)
        assert len(ax.lines) == len(expected)
        for line, (a, b) in zip(ax.lines, expected):
            np.testing.assert_array_equal(np.column_stack(line.get_data()), vertices[[a, b], :2])
        fig.canvas.draw()
        # A unit X displacement and unit Y displacement have the same screen length.
        origin, x, y = ax.transData.transform([(0, 0), (1, 0), (0, 1)])
        assert np.linalg.norm(x - origin) == pytest.approx(np.linalg.norm(y - origin))
        for column, limits in enumerate((ax.get_xlim(), ax.get_ylim())):
            assert limits[0] < vertices[:, column].min() <= vertices[:, column].max() < limits[1]
    finally:
        plt.close(fig)


@pytest.mark.parametrize('volume,coordinates', [(False, False), (False, True), (True, False), (True, True)])
def test_disabled_sections_are_removed_from_actual_svg_layout(volume, coordinates):
    result, svg = svg_result(Tetrahedron(), show_volume_ratios=volume, show_coordinates=coordinates)
    panels = {element.attrib['id'] for element in svg.iter()
              if element.attrib.get('id', '').startswith('solid-')}
    expected = {'solid-structure', 'solid-xy-projection'}
    if volume:
        expected.add('solid-volume-comparison')
    if coordinates:
        expected.add('solid-properties')
    assert panels == expected
    if not (volume or coordinates):
        assert float(svg.attrib['height'].removesuffix('pt')) < 500
    projection = next(element for element in svg.iter() if element.attrib.get('id') == 'solid-xy-projection')
    assert sum(element.attrib.get('id', '').startswith('projected-edge-')
               for element in projection.iter()) == 6
    assert len(result['metadata']['panels']) == len(expected)


@pytest.mark.parametrize('power', [200, 5000])
def test_huge_translation_preserves_exact_metadata_and_bounds_svg_text(tmp_path, power):
    model = Octahedron([Q.from_xyz(*point) for point in
                        [(4, 0, 0), (-2, 0, 0), (0, 2, 0),
                         (0, -2, 0), (0, 0, 2), (0, 0, -2)]])
    before, svg_before = svg_result(model)
    original_vertices, _ = _polyhedron_frame(model)
    shift = Q(10**power, 0, 0, 0)
    model.vertices[:] = [vertex.add(shift) for vertex in model.vertices]
    set_config({'output_dir': str(tmp_path / 'translated')})
    after, svg_after = svg_result(model)
    translated_vertices, _ = _polyhedron_frame(model)
    np.testing.assert_array_equal(translated_vertices, original_vertices)
    assert after['metadata']['volume'] == before['metadata']['volume']
    assert after['metadata']['vertices_quadray_exact'] == tuple(v.as_tuple() for v in model.vertices)
    assert after['metadata']['origin_quadray_exact'] == model.vertices[0].as_tuple()
    assert after['metadata']['geometry_analysis']['vertex_centroid_exact'] != after['metadata']['geometry_analysis']['volume_centroid_exact']
    for dimension in ('width', 'height'):
        value = float(svg_after.attrib[dimension].removesuffix('pt'))
        baseline = float(svg_before.attrib[dimension].removesuffix('pt'))
        assert value == pytest.approx(baseline, rel=0.02)
    texts = [''.join(element.itertext()) for element in svg_after.iter()
             if element.tag.endswith('}text')]
    assert max(map(len, texts)) < 120


def test_hidden_geometry_layers_remove_actual_projected_artists():
    _, svg = svg_result(Cube(), show_coordinates=False, show_volume_ratios=False,
                        show_edges=False, show_faces=False, show_vertices=False)
    projection = next(element for element in svg.iter() if element.attrib.get('id') == 'solid-xy-projection')
    assert not any(element.attrib.get('id', '').startswith(('projected-edge-', 'PathCollection_'))
                   for element in projection.iter())


def test_mid_save_filesystem_failure_closes_figure_and_preserves_prior_consumer_figure(tmp_path):
    plt = pytest.importorskip('matplotlib.pyplot')
    consumer_figure = plt.figure()
    try:
        figures = plt.get_fignums()
        set_config({'formats': ['png', 'svg']})
        (tmp_path / 'tetrahedron_graphical_abstract.svg').mkdir()
        with pytest.raises(OSError):
            plot_polyhedron_graphical_abstract(Tetrahedron())
        assert plt.get_fignums() == figures
        assert plt.imread(tmp_path / 'tetrahedron_graphical_abstract.png').std() > 0
    finally:
        plt.close(consumer_figure)


def test_invalid_camera_during_drawing_closes_figure():
    plt = pytest.importorskip('matplotlib.pyplot')
    figures = plt.get_fignums()
    with pytest.raises((TypeError, ValueError)):
        plot_polyhedron_graphical_abstract(Tetrahedron(), elevation='not an angle')
    assert plt.get_fignums() == figures
