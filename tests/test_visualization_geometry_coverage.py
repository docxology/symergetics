"""Numerical rendered-geometry contracts, not artifact-existence coverage."""

import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest

from symergetics.core.coordinates import QuadrayCoordinate as Q
from symergetics.geometry.polyhedra import Cube, Cuboctahedron, Octahedron, Tetrahedron
from symergetics.visualization import set_config
from symergetics.visualization.geometry import (
    _draw_polyhedron, _fit_3d, _polyhedron_frame,
    plot_polyhedron, plot_polyhedron_3d, plot_polyhedron_graphical_abstract,
    plot_polyhedron_wireframe,
)


@pytest.fixture(autouse=True)
def small_output(tmp_path):
    set_config(dict(output_dir=str(tmp_path), organize_by_type=False,
                    include_timestamps=False, figure_size=(3, 3), dpi=50,
                    formats=['png']))


def plotly_artifact(result):
    """Decode actual public HTML output, including binary Plotly arrays."""
    html = Path(result['files'][0]).read_text()
    start = html.index('[', html.rindex('Plotly.newPlot('))
    decoder = json.JSONDecoder()
    traces, end = decoder.raw_decode(html[start:])
    layout, _ = decoder.raw_decode(html[start + end:].lstrip(' ,\n'))
    return traces, layout


def array(value):
    if isinstance(value, dict):
        return np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
    return np.asarray(value)


def xyz(trace):
    return np.column_stack([array(trace[axis]) for axis in 'xyz'])


def triangle_area(triangles):
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2


@pytest.mark.parametrize('factory,count,area', [
    (Tetrahedron, 4, 8 * math.sqrt(3)), (Cube, 12, 24),
    (Octahedron, 8, 16 * math.sqrt(3)), (Cuboctahedron, 20, 48 + 16 * math.sqrt(3)),
])
def test_plotly_and_matplotlib_emit_complete_nondegenerate_surface(factory, count, area):
    pytest.importorskip('plotly')
    plt = pytest.importorskip('matplotlib.pyplot')
    model = factory()
    traces, _ = plotly_artifact(plot_polyhedron(model, backend='plotly'))
    mesh = next(t for t in traces if t['type'] == 'mesh3d')
    indices = np.column_stack([array(mesh[key]) for key in 'ijk']).astype(int)
    triangles = xyz(mesh)[indices]
    assert triangles.shape == (count, 3, 3)
    assert triangle_area(triangles) == pytest.approx(area)
    assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) > 0)
    # Inspect the real Matplotlib collection used by static plots AND animation.
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection='3d')
        vertices, _ = _polyhedron_frame(model)
        _draw_polyhedron(ax, model, vertices, show_edges=False, show_vertices=False)
        actual = ax.collections[0]._vec[:3].T.reshape(-1, 3, 3)
        np.testing.assert_allclose(actual, triangles)
        assert triangle_area(actual) == pytest.approx(area)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('edges,faces,vertices', [(False, False, False),
    (True, False, False), (False, True, False), (False, False, True)])
def test_public_plotly_visibility_changes_real_traces(edges, faces, vertices):
    pytest.importorskip('plotly')
    traces, _ = plotly_artifact(plot_polyhedron(
        Tetrahedron(), backend='plotly', show_edges=edges,
        show_faces=faces, show_vertices=vertices))
    assert sum(t['type'] == 'mesh3d' for t in traces) == int(faces)
    assert sum(t.get('mode') == 'markers' for t in traces) == int(vertices)
    assert sum(t.get('mode') == 'lines' for t in traces) == (6 if edges else 0)


def test_requested_plotly_angles_move_actual_camera():
    pytest.importorskip('plotly')
    _, layout = plotly_artifact(plot_polyhedron_3d(
        Cube(), backend='plotly', elevation=30, azimuth=90,
        show_wireframe=False, show_surface=True))
    eye = layout['scene']['camera']['eye']
    assert eye['x'] == pytest.approx(0, abs=1e-14)
    assert eye['y'] == pytest.approx(2.5 * math.sqrt(3) / 2)
    assert eye['z'] == pytest.approx(1.25)
    _, other = plotly_artifact(plot_polyhedron_3d(
        Cube(), backend='plotly', elevation=-20, azimuth=0))
    assert other['scene']['camera']['eye'] != eye


def test_skew_and_huge_translation_fit_the_actual_relative_mesh():
    pytest.importorskip('plotly')
    model = Tetrahedron([Q(0, 0, 0, 0), Q(10, 0, 0, 10),
                         Q(1, 1, 0, 0), Q(1, 0, 1, 0)])
    before, _ = plotly_artifact(plot_polyhedron(model, backend='plotly'))
    anchor = Q(2**60, 0, 0, 0)
    model.vertices[:] = [vertex.add(anchor) for vertex in model.vertices]
    result = plot_polyhedron(model, backend='plotly')
    after, layout = plotly_artifact(result)
    mesh = next(t for t in after if t['type'] == 'mesh3d')
    points = xyz(mesh)
    np.testing.assert_array_equal(points, xyz(next(t for t in before if t['type'] == 'mesh3d')))
    np.testing.assert_array_equal(points.max(axis=0), [20, 2, 2])
    scene = layout['scene']
    assert scene['aspectmode'] == 'cube'
    spans = []
    for column, axis in enumerate('xyz'):
        low, high = scene[f'{axis}axis']['range']
        assert low < points[:, column].min() <= points[:, column].max() < high
        assert (low + high) / 2 == pytest.approx((points[:, column].min() + points[:, column].max()) / 2)
        assert 'anchor-relative' in scene[f'{axis}axis']['title']['text']
        spans.append(high - low)
    assert len(set(spans)) == 1
    assert result['metadata']['origin_quadray_exact'] == anchor.as_tuple()
    assert result['metadata']['volume'] == 5


def test_coincident_vertices_do_not_reindex_declared_plotly_faces():
    pytest.importorskip('plotly')
    model = Tetrahedron()
    model.vertices[1] = model.vertices[0]
    traces, _ = plotly_artifact(plot_polyhedron(model, backend='plotly'))
    mesh = next(t for t in traces if t['type'] == 'mesh3d')
    actual = list(zip(*(array(mesh[key]).tolist() for key in 'ijk')))
    assert actual == list(model.triangulated_faces())
    assert any(0 in triangle for triangle in actual)
    assert any(1 in triangle for triangle in actual)


def test_collapsed_matplotlib_frame_has_positive_metric_extent():
    plt = pytest.importorskip('matplotlib.pyplot')
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection='3d')
        _fit_3d(ax, [(2, 3, 4)] * 4, relative=True)
        assert ax.get_xlim() == (1, 3)
        assert ax.get_ylim() == (2, 4)
        assert ax.get_zlim() == (3, 5)
        np.testing.assert_allclose(ax.get_box_aspect() / ax.get_box_aspect()[0], [1, 1, 1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize('function', [plot_polyhedron_3d,
    plot_polyhedron_graphical_abstract, plot_polyhedron_wireframe])
def test_specialized_geometry_rejects_unsupported_backend(function):
    with pytest.raises(ValueError):
        function(Tetrahedron(), backend='ascii')


def test_advanced_contacts_and_cuboctahedral_edges_are_metric_geometry():
    plt = pytest.importorskip('matplotlib.pyplot')
    from symergetics.visualization.advanced import (
        _create_sphere_packing_panel, _create_vector_equilibrium_panel,
    )
    fig, axes = plt.subplots(1, 2)
    try:
        _create_sphere_packing_panel(axes[0])
        contacts = np.column_stack(axes[0].child_axes[0].collections[0]._offsets3d)
        assert contacts.shape == (13, 3)
        np.testing.assert_array_equal(contacts[0], [0, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(contacts[1:], axis=1), 2)
        distances = np.linalg.norm(contacts[:, None] - contacts[None, :], axis=2)
        assert distances[np.triu_indices(13, 1)].min() == pytest.approx(2)

        _create_vector_equilibrium_panel(axes[1])
        view = axes[1].child_axes[0]
        assert len(view.lines) == 24
        edges = [np.column_stack(line.get_data_3d()) for line in view.lines]
        lengths = [np.linalg.norm(edge[1] - edge[0]) for edge in edges]
        np.testing.assert_allclose(lengths, math.sqrt(8))
        points = {tuple(point) for edge in edges for point in edge}
        assert len(points) == 12
        assert all(np.linalg.norm(point) == pytest.approx(math.sqrt(8)) for point in points)
        assert 'V = 20 IVM' in '\n'.join(t.get_text() for t in axes[1].texts)
    finally:
        plt.close(fig)


def test_advanced_models_share_physical_scale_and_cluster_counts():
    plt = pytest.importorskip('matplotlib.pyplot')
    from symergetics.visualization.advanced import (
        _create_geometric_ratios_panel, _create_synergetics_sphere_frequency_panel,
    )
    fig, axes = plt.subplots(1, 2)
    try:
        _create_geometric_ratios_panel(axes[0])
        view = axes[0].child_axes[0]
        # Six tetrahedron, twelve cube, twelve octahedron edges, without rescaling.
        lengths = [np.linalg.norm(np.diff(np.column_stack(line.get_data_3d()), axis=0))
                   for line in view.lines]
        np.testing.assert_allclose(lengths[:6], math.sqrt(8))
        np.testing.assert_allclose(lengths[6:18], 2)
        np.testing.assert_allclose(lengths[18:], math.sqrt(8))
        assert len(lengths) == 30
        _create_synergetics_sphere_frequency_panel(axes[1])
        chart = axes[1].child_axes[0]
        layers, counts = chart.lines[0].get_data()
        expected = [math.comb(int(n) + 2, 3) for n in layers]
        np.testing.assert_array_equal(counts, expected)
    finally:
        plt.close(fig)


def test_quadray_panel_draws_actual_basis_and_gauge_origin():
    plt = pytest.importorskip('matplotlib.pyplot')
    from symergetics.visualization.advanced import _create_quadray_coordinate_panel
    fig, ax = plt.subplots()
    try:
        _create_quadray_coordinate_panel(ax)
        view = ax.child_axes[0]
        endpoints = [np.column_stack(line.get_data_3d())[-1] for line in view.lines]
        np.testing.assert_array_equal(endpoints, [[1, 1, 1], [-1, 1, -1],
                                                 [-1, -1, 1], [1, -1, -1]])
        np.testing.assert_array_equal(np.sum(endpoints, axis=0), [0, 0, 0])
        example_points = np.vstack([np.column_stack(c._offsets3d) for c in view.collections])
        assert any(np.array_equal(point, [0, 0, 0]) for point in example_points)
        equations = '\n'.join(text.get_text() for text in ax.texts)
        assert 'x = a − b − c + d' in equations
        assert 'No sum-to-one constraint' in equations
    finally:
        plt.close(fig)


def test_graphical_properties_present_live_area_and_distinct_exact_centroids():
    plt = pytest.importorskip('matplotlib.pyplot')
    set_config({'formats': ['svg']})
    # Unequal x tips: vertex mean x=1/3, uniform solid centroid x=1/2.
    model = Octahedron([Q.from_xyz(*point) for point in
                        [(4, 0, 0), (-2, 0, 0), (0, 2, 0),
                         (0, -2, 0), (0, 0, 2), (0, 0, -2)]])
    with plt.rc_context({'svg.fonttype': 'none'}):
        result = plot_polyhedron_graphical_abstract(model)
    analysis = result['metadata']['geometry_analysis']
    assert analysis['vertex_centroid_exact'] != analysis['volume_centroid_exact']
    assert analysis['affine_rank'] == 3
    assert analysis['closed'] is True
    assert analysis['surface_area'] == pytest.approx(model.surface_area())
    import xml.etree.ElementTree as ET
    text = '\n'.join(ET.parse(result['files'][0]).getroot().itertext())
    assert 'Exact vertex centroid (quadray)' in text
    assert 'Exact solid centroid (quadray)' in text
    assert 'Surface area:' in text
    assert 'closed topology: True' in text
