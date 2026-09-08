"""Object attribute tables and geometry helpers shared by physics and render.

Geometry is defined once here so collision shapes (pybullet) and rendered meshes
(bpy) stay aligned. A size scale `s` is the sphere radius; cube and cylinder are
scaled down so their enclosed volume matches the sphere, giving all shapes the
same apparent visual size:
    sphere:   radius s
    cube:     half-extent s * 0.80   (≈ equal volume to sphere)
    cylinder: radius s * 0.87, half-height s * 0.87   (≈ equal volume)
All shapes rest with their center at z = effective_half_extent on the ground.
"""

# Per-shape scale applied to the size_scale parameter so all shapes look the
# same size as a sphere of that radius.
_SHAPE_SCALE = {
    'sphere':   1.00,
    'cube':     0.80,
    'cylinder': 0.87,
}


def color_to_rgba(rgb255):
    """[0-255] RGB list -> linear-ish [0-1] RGBA (alpha 1)."""
    return [c / 255.0 for c in rgb255] + [1.0]


def rest_height(shape, size_scale):
    """Center z-height for a shape of given size_scale resting on the ground."""
    return size_scale * _SHAPE_SCALE[shape]


def half_extents(shape, s):
    """Half-extents / radius used by the collision body and the rendered mesh.

    Clips rendered before 2026-08-13 did not apply _SHAPE_SCALE to the mesh; use
    render_half_extents(..., legacy=True) for those.
    """
    sc = s * _SHAPE_SCALE[shape]
    if shape == 'cube':
        return ('box', (sc, sc, sc))
    if shape == 'sphere':
        return ('sphere', sc)
    if shape == 'cylinder':
        return ('cylinder', (sc, sc))   # (radius, half_height)
    raise ValueError(f'unknown shape: {shape}')


def render_half_extents(shape, s, legacy=False):
    """[x, y, z] half-extents of the rendered mesh, in world units.

    `legacy=True` returns the geometry used by clips rendered before the
    2026-08-13 fix, when Renderer.add_objects applied no _SHAPE_SCALE. Only
    cubes and cylinders differ; spheres are identical either way.
    """
    if legacy:
        return [float(s), float(s), float(s)]
    kind, dim = half_extents(shape, s)
    if kind == 'sphere':
        return [float(dim)] * 3
    if kind == 'box':
        return [float(d) for d in dim]
    radius, half_h = dim
    return [float(radius), float(radius), float(half_h)]


class Sampler:
    """Draws object attributes from the `properties` block of a config."""

    def __init__(self, props, rng):
        self.shapes = list(props['shapes'])
        self.materials = list(props['materials'])
        self.colors = dict(props['colors'])
        self.sizes = dict(props['sizes'])
        self.rng = rng

    def sample(self, key, value):
        return self.rng.choice(value)

    def shape(self):
        return self.shapes[self.rng.integers(len(self.shapes))]

    def material(self):
        return self.materials[self.rng.integers(len(self.materials))]

    def color(self):
        name = list(self.colors)[self.rng.integers(len(self.colors))]
        return name, self.colors[name]

    def size(self):
        name = list(self.sizes)[self.rng.integers(len(self.sizes))]
        return name, self.sizes[name]
