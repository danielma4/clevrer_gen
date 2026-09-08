"""Camera reconstruction and projection, independent of any rendered scene.

The generator inherits CLEVR's per-scene camera jitter: _build_scene offsets
_CAM_LOC by rng.uniform(-0.5, 0.5) per axis, so every clip has a *different*
camera (measured over 5000 clips: 5000 distinct positions, 1.0 peak-to-peak per
axis). Assuming a single dataset-wide matrix misprojects the play area by ~12 px
on average against a ~37 px ball, so the matrix belongs in each annotation.

The camera is set once per clip and never keyframed, though, so it is static
within a clip: one matrix per clip, not one per frame.

Because the jitter is a pure function of (seed, index), this module replays the
draw and rebuilds the camera without constructing a bpy scene, which is what
makes backfilling 27k annotations and projecting pixels on CPU practical. Draw
order must match _build_scene exactly: the three area lights consume three
uniforms each, then the camera consumes three.
"""
import numpy as np

from .render import _CAM_LOC, _LIGHTS


def camera_location(seed, light_jitter=1.0, camera_jitter=0.5):
    """Replay the generator's jitter draws and return the camera location.

    Rounded through float32 because Blender stores object.location at that
    precision; staying in float64 disagrees with the render by ~1e-7 units.
    """
    rng = np.random.default_rng(seed)
    for _, loc, _, _ in _LIGHTS:
        tuple(c + rng.uniform(-light_jitter, light_jitter) for c in loc)
    loc = tuple(c + rng.uniform(-camera_jitter, camera_jitter) for c in _CAM_LOC)
    return np.asarray(loc, dtype=np.float32).astype(np.float64)


def look_at_matrix(location, target=(0.0, 0.0, 0.0)):
    """4x4 camera-to-world matrix for a camera at `location` aimed at `target`.

    Delegates to mathutils' to_track_quat so the basis matches _look_at bit for
    bit; hand-rolling the track-quat convention is how a handedness flip sneaks
    in and quietly corrupts every motion label.
    """
    from mathutils import Matrix, Vector

    loc = Vector(tuple(float(c) for c in location))
    direction = Vector(tuple(float(c) for c in target)) - loc
    rot = direction.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
    return np.array(Matrix.Translation(loc) @ rot, dtype=np.float64)


def intrinsics(width, height, lens=45.0, sensor_width=36.0):
    """Pinhole intrinsics for Blender's default AUTO sensor fit.

    AUTO fits the sensor to the larger render dimension, so focal length in
    pixels is driven by width whenever width >= height (480x320 here).
    """
    f = max(width, height) * lens / sensor_width
    return {
        'focal_length_mm': float(lens),
        'sensor_width_mm': float(sensor_width),
        'sensor_fit': 'AUTO',
        'resolution': [int(width), int(height)],
        'fx': float(f), 'fy': float(f),
        'cx': width / 2.0, 'cy': height / 2.0,
        'K': [[float(f), 0.0, width / 2.0],
              [0.0, float(f), height / 2.0],
              [0.0, 0.0, 1.0]],
    }


class Camera:
    """Static per-clip pinhole camera in Blender's -Z forward / +Y up basis."""

    def __init__(self, seed, width, height, lens=45.0, sensor_width=36.0,
                 light_jitter=1.0, camera_jitter=0.5, target=(0.0, 0.0, 0.0)):
        self.width, self.height = int(width), int(height)
        self.location = camera_location(seed, light_jitter, camera_jitter)
        self.cam_to_world = look_at_matrix(self.location, target)
        self.world_to_cam = np.linalg.inv(self.cam_to_world)
        self.intr = intrinsics(width, height, lens, sensor_width)
        self.R = self.cam_to_world[:3, :3]      # camera basis in world coords
        self.fx, self.fy = self.intr['fx'], self.intr['fy']
        self.cx, self.cy = self.intr['cx'], self.intr['cy']

    def to_camera(self, points):
        """World [..., 3] -> camera-space [..., 3] (looking down -Z)."""
        p = np.asarray(points, dtype=np.float64)
        return (p - self.location) @ self.R

    def project(self, points):
        """World [..., 3] -> (pixels [..., 2], depth [...]).

        Pixel coords are continuous with the origin at the image's top-left
        corner, matching the row order of the rendered passes and masks. `depth`
        is measured along the view axis and is positive in front of the camera.
        """
        pc = self.to_camera(points)
        depth = -pc[..., 2]
        safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
        px = self.cx + self.fx * (pc[..., 0] / safe)
        py = self.cy - self.fy * (pc[..., 1] / safe)
        return np.stack([px, py], axis=-1), depth

    def pixel_rays(self, px, py):
        """Pixel coords -> unit world-space ray directions.

        Normalised so a ray parameter is a true distance, letting the shape
        intersectors compare hits across objects directly for z-ordering.
        """
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
        d_cam = np.stack([(px - self.cx) / self.fx,
                          -(py - self.cy) / self.fy,
                          -np.ones_like(px)], axis=-1)
        d = d_cam @ self.R.T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def to_dict(self):
        """Serialisable intrinsics + extrinsics for an annotation JSON."""
        from mathutils import Matrix
        q = Matrix(self.cam_to_world.tolist()).to_quaternion()
        d = dict(self.intr)
        d.update({
            'location': self.location.tolist(),
            'rotation_quaternion_wxyz': [q.w, q.x, q.y, q.z],
            'cam_to_world': self.cam_to_world.tolist(),
            'world_to_cam': self.world_to_cam.tolist(),
            'static_within_clip': True,
            'convention': 'blender: camera looks along -Z with +Y up; '
                          'pixel origin at image top-left',
        })
        return d
